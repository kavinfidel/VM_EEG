"""
train.py
========
Subject-specific training loop with:
  - K-fold cross-validation (stratified)
  - Early stopping on validation loss
  - Learning rate scheduling (OneCycleLR)
  - Returns per-fold and aggregate metrics

Metrics returned:
  accuracy, balanced_accuracy, kappa (Cohen's κ),
  per-class F1, confusion matrix.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
)

from typing import Optional
from models import EEGGraphNet
from dataset import EEGGraphDataset


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    n_classes: int) -> dict:
    return {
        'accuracy':          accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'kappa':             cohen_kappa_score(y_true, y_pred),
        'f1_macro':          f1_score(y_true, y_pred, average='macro',
                                       zero_division=0),
        'f1_per_class':      f1_score(y_true, y_pred, average=None,
                                       zero_division=0).tolist(),
        'confusion_matrix':  confusion_matrix(y_true, y_pred,
                                               labels=list(range(n_classes))).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    train_graphs: list[Data],
    val_graphs:   list[Data],
    model_kwargs: dict,
    n_classes:    int,
    epochs:       int = 100,
    batch_size:   int = 32,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int = 20,
    device:       Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    verbose:      bool = False,
) -> tuple[EEGGraphNet, dict, dict]:
    """
    Train for one fold. Returns (model, train_metrics, val_metrics).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = EEGGraphDataset(train_graphs)
    val_ds   = EEGGraphDataset(val_graphs)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=0)

    model = EEGGraphNet(n_classes=n_classes, **model_kwargs).to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ── validate ──
        model.eval()
        val_loss = 0.0
        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                logits = model(batch)
                loss   = criterion(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                preds  = logits.argmax(dim=1).cpu().numpy()
                y_pred_v.extend(preds.tolist())
                y_true_v.extend(batch.y.cpu().numpy().tolist())

        val_loss /= len(val_ds)

        if verbose and epoch % 20 == 0:
            acc = accuracy_score(y_true_v, y_pred_v)
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={acc:.3f}")

        # ── early stopping ──
        if val_loss < best_val_loss - 1e-5:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}.")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── final evaluation ──
    model.eval()

    def eval_loader(loader):
        yt, yp = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds = model(batch).argmax(dim=1).cpu().numpy()
                yp.extend(preds.tolist())
                yt.extend(batch.y.cpu().numpy().tolist())
        return np.array(yt), np.array(yp)

    y_true_tr, y_pred_tr = eval_loader(train_loader)
    y_true_vl, y_pred_vl = eval_loader(val_loader)

    return (
        model,
        compute_metrics(y_true_tr, y_pred_tr, n_classes),
        compute_metrics(y_true_vl, y_pred_vl, n_classes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated training for one subject
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_subject(
    graphs:       list[Data],
    labels:       np.ndarray,
    model_kwargs: dict,
    n_classes:    int,
    n_splits:     int = 5,
    epochs:       int = 100,
    batch_size:   int = 32,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int = 20,
    device:       Optional[str] = None,
    use_class_weights: bool = True,
    verbose:      bool = True,
) -> dict:
    """
    Stratified K-Fold CV for one subject's graph list.

    Returns aggregated metrics dict with mean/std over folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    # Class weights for imbalance handling
    class_weights = None
    if use_class_weights:
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        w = w / w.sum() * n_classes
        class_weights = torch.tensor(w, dtype=torch.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(graphs, labels), 1):
        if verbose:
            print(f"  Fold {fold}/{n_splits}")
        train_g = [graphs[i] for i in train_idx]
        val_g   = [graphs[i] for i in val_idx]

        _, _, val_m = train_one_fold(
            train_graphs=train_g,
            val_graphs=val_g,
            model_kwargs=model_kwargs,
            n_classes=n_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            class_weights=class_weights,
            verbose=verbose,
        )
        fold_metrics.append(val_m)
        if verbose:
            print(f"    → acc={val_m['accuracy']:.3f} | κ={val_m['kappa']:.3f}")

    # Aggregate
    keys = ['accuracy', 'balanced_accuracy', 'kappa', 'f1_macro']
    result = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        result[f'{k}_mean'] = float(np.mean(vals))
        result[f'{k}_std']  = float(np.std(vals))

    result['n_folds']      = n_splits
    result['fold_metrics'] = fold_metrics
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Train on all data, evaluate on held-out test set (block-2 last trial)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_test(
    train_graphs: list[Data],
    test_graphs:  list[Data],
    train_labels: np.ndarray,
    model_kwargs: dict,
    n_classes:    int,
    epochs:       int = 150,
    batch_size:   int = 32,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int = 30,
    device:       Optional[str] = None,
    use_class_weights: bool = True,
    verbose:      bool = True,
) -> tuple[EEGGraphNet, dict]:
    """
    Train on full training set, report metrics on the dedicated test set.
    Uses an 10% validation split (from training) for early stopping only.
    """
    from sklearn.model_selection import train_test_split

    # Internal val split (stratified 10%) for early stopping
    idx = np.arange(len(train_graphs))
    tr_idx, v_idx = train_test_split(idx, test_size=0.1,
                                     stratify=train_labels, random_state=42)
    tr_g = [train_graphs[i] for i in tr_idx]
    v_g  = [train_graphs[i] for i in v_idx]

    class_weights = None
    if use_class_weights:
        counts = np.bincount(train_labels, minlength=n_classes).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        w = w / w.sum() * n_classes
        class_weights = torch.tensor(w, dtype=torch.float32)

    model, _, _ = train_one_fold(
        train_graphs=tr_g, val_graphs=v_g,
        model_kwargs=model_kwargs, n_classes=n_classes,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, patience=patience,
        device=device, class_weights=class_weights, verbose=verbose,
    )

    # Evaluate on held-out test set
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)

    test_ds     = EEGGraphDataset(test_graphs)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch  = batch.to(device)
            preds  = model(batch).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(batch.y.cpu().numpy().tolist())

    test_metrics = compute_metrics(np.array(y_true), np.array(y_pred), n_classes)
    if verbose:
        print(f"  Test → acc={test_metrics['accuracy']:.3f} | "
              f"κ={test_metrics['kappa']:.3f} | "
              f"bal_acc={test_metrics['balanced_accuracy']:.3f}")

    return model, test_metrics
