"""
train.py
========
Subject-specific training loop with:
  - K-fold cross-validation (stratified)
  - Early stopping on validation loss
  - Learning rate scheduling (OneCycleLR)
  - tqdm progress bars: one bar per epoch, one bar per fold/subject
  - MPS / CUDA / CPU device resolution

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
from tqdm.auto import tqdm

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
# Device resolution — CUDA > MPS > CPU
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(device: Optional[str] = None) -> str:
    """
    Returns the best available device if device=None.
    Priority: CUDA > MPS (Apple Silicon) > CPU.

    You can force a backend by passing device='cpu', 'cuda', or 'mps'.
    In GNNConfig set device='mps' to use Apple Silicon GPU.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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
    train_graphs:  list[Data],
    val_graphs:    list[Data],
    model_kwargs:  dict,
    n_classes:     int,
    epochs:        int = 100,
    batch_size:    int = 32,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    patience:      int = 20,
    device:        Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
    verbose:       bool = True,
    pbar_desc:     str = 'Training',
) -> tuple[EEGGraphNet, dict, dict]:
    """
    Train for one fold. Returns (model, train_metrics, val_metrics).

    A tqdm bar runs every epoch showing tr_loss / vl_loss / vl_acc / patience.
    Early-stop information is always printed via tqdm.write (won't break bars).
    """
    device = resolve_device(device)

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

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None
    stopped_epoch    = epochs

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=pbar_desc,
        unit='ep',
        dynamic_ncols=True,
        leave=False,       # collapses after done so fold/subject bars stay clean
    )

    for epoch in epoch_bar:
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch  = batch.to(device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_ds)

        # ── validate ────────────────────────────────────────────────────────
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
        val_acc   = accuracy_score(y_true_v, y_pred_v)

        epoch_bar.set_postfix(
            tr=f'{train_loss:.3f}',
            vl=f'{val_loss:.3f}',
            acc=f'{val_acc:.3f}',
            p=f'{patience_counter}/{patience}',
        )

        # ── early stopping ──────────────────────────────────────────────────
        # min_delta=1e-4: ignore improvements smaller than this to avoid
        # patience firing on pure numerical jitter (1e-5 was too tight).
        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch
                epoch_bar.close()
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    stop_str = (f'early stop @ep{stopped_epoch}'
                if stopped_epoch < epochs else f'completed {epochs} ep')
    tqdm.write(f'    {pbar_desc} | {stop_str} | '
               f'best_vl={best_val_loss:.4f} | [{device}]')

    # ── final evaluation ────────────────────────────────────────────────────
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
    subject_id:   str = '',
) -> dict:
    """
    Stratified K-Fold CV for one subject's graph list.
    Returns aggregated metrics dict with mean/std over folds.
    """
    device = resolve_device(device)
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    class_weights = None
    if use_class_weights:
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        counts[counts == 0] = 1.0
        w = 1.0 / counts
        w = w / w.sum() * n_classes
        class_weights = torch.tensor(w, dtype=torch.float32)

    fold_bar = tqdm(
        enumerate(skf.split(graphs, labels), 1),
        total=n_splits,
        desc=f'{subject_id} folds',
        leave=True,
    )

    for fold, (train_idx, val_idx) in fold_bar:
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
            pbar_desc=f'{subject_id} fold {fold}/{n_splits}',
        )
        fold_metrics.append(val_m)
        fold_bar.set_postfix(
            acc=f"{val_m['accuracy']:.3f}",
            kappa=f"{val_m['kappa']:.3f}",
        )

    keys   = ['accuracy', 'balanced_accuracy', 'kappa', 'f1_macro']
    result = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        result[f'{k}_mean'] = float(np.mean(vals))
        result[f'{k}_std']  = float(np.std(vals))

    result['n_folds']      = n_splits
    result['fold_metrics'] = fold_metrics
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Train on all data, evaluate on held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def train_and_test(
    train_graphs:  list[Data],
    test_graphs:   list[Data],
    train_labels:  np.ndarray,
    model_kwargs:  dict,
    n_classes:     int,
    epochs:        int = 150,
    batch_size:    int = 32,
    lr:            float = 1e-3,
    weight_decay:  float = 1e-4,
    patience:      int = 30,
    device:        Optional[str] = None,
    use_class_weights: bool = True,
    verbose:       bool = True,
    subject_id:    str = '',
) -> tuple[EEGGraphNet, dict]:
    """
    Train on full training set, evaluate on dedicated test set.
    Uses a stratified 10% val split for early stopping only.
    """
    from sklearn.model_selection import train_test_split

    device = resolve_device(device)

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
        train_graphs=tr_g,
        val_graphs=v_g,
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
        pbar_desc=f'{subject_id}' if subject_id else 'Training',
    )

    # ── test set evaluation ──────────────────────────────────────────────────
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

    tqdm.write(
        f'  ► {subject_id or "Subject"} TEST | '
        f'acc={test_metrics["accuracy"]:.3f} | '
        f'κ={test_metrics["kappa"]:.3f} | '
        f'bal_acc={test_metrics["balanced_accuracy"]:.3f}'
    )

    return model, test_metrics
