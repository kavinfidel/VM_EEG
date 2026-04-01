"""
pipeline.py
===========
Top-level orchestration: ties preprocessing → feature extraction →
graph construction → GNN training into a single callable.

This is what the sweep notebook calls.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Literal, Optional

from features import (
    compute_classical_features,
    train_autoencoder,
    extract_ae_features,
)
from graph import build_graphs_for_subject
from train import cross_validate_subject, train_and_test


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class GNNConfig:
    # ── Feature strategy ──────────────────────────────────────────────────
    feature_type:   Literal['classical', 'autoencoder'] = 'classical'

    # Classical feature options
    symmetric_pairs: Optional[list[tuple[int, int]]] = None  # for RASM

    # Autoencoder options
    ae_latent_dim:  int   = 32
    ae_epochs:      int   = 50
    ae_batch_size:  int   = 512
    ae_lr:          float = 1e-3

    # ── Graph construction ────────────────────────────────────────────────
    adj_method:         Literal['plv', 'pearson'] = 'plv'
    adj_threshold:      float = 0.3
    keep_top_k_percent: Optional[float] = None   # e.g. 20.0 = top 20%

    # ── GNN model ─────────────────────────────────────────────────────────
    arch:       Literal['gcn', 'graphsage', 'gat'] = 'graphsage'
    hidden_dim: int   = 64
    n_layers:   int   = 3
    dropout:    float = 0.4
    # GraphSAGE-specific
    sage_aggr:  str   = 'mean'   # 'mean' | 'max'
    # GAT-specific
    gat_heads:  int   = 4

    # ── Training ──────────────────────────────────────────────────────────
    epochs:       int   = 100
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    patience:     int   = 20
    n_cv_folds:   int   = 5
    use_class_weights: bool = True

    # ── Misc ──────────────────────────────────────────────────────────────
    fs:      float = 250.0
    device:  Optional[str] = None
    verbose: bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    windows: np.ndarray,    # (N, C, T)
    cfg: GNNConfig,
    ae_model=None,          # pre-trained AE (optional; will train if None)
) -> tuple[np.ndarray, object]:
    """
    Returns (node_features, ae_model_or_None).
    node_features: (N, C, F)
    """
    if cfg.feature_type == 'classical':
        if cfg.verbose:
            print("  Extracting classical features (DE + RASM + Hjorth) …")
        feats = compute_classical_features(
            windows, fs=cfg.fs, symmetric_pairs=cfg.symmetric_pairs
        )
        return feats, None

    elif cfg.feature_type == 'autoencoder':
        if ae_model is None:
            if cfg.verbose:
                print("  Training channel autoencoder …")
            ae_model = train_autoencoder(
                windows,
                latent_dim=cfg.ae_latent_dim,
                epochs=cfg.ae_epochs,
                batch_size=cfg.ae_batch_size,
                lr=cfg.ae_lr,
                device=cfg.device,
                verbose=cfg.verbose,
            )
        if cfg.verbose:
            print("  Extracting AE latent features …")
        feats = extract_ae_features(ae_model, windows, device=cfg.device)
        return feats, ae_model

    else:
        raise ValueError(f"Unknown feature_type: {cfg.feature_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Model kwargs builder (handles arch-specific args)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_kwargs(cfg: GNNConfig, in_features: int) -> dict:
    kwargs = dict(
        in_features=in_features,
        arch=cfg.arch,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    )
    if cfg.arch == 'graphsage':
        kwargs['aggr'] = cfg.sage_aggr
    elif cfg.arch == 'gat':
        kwargs['heads']    = cfg.gat_heads
        kwargs['edge_dim'] = 1
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject CV pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_subject_cv(
    subject_id: str,
    windows:    np.ndarray,   # (N, C, T) — training windows
    labels:     np.ndarray,   # (N,)
    cfg:        GNNConfig,
    n_classes:  int,
    ae_model=None,
) -> dict:
    """
    Full CV pipeline for one subject.
    Returns metrics dict from cross_validate_subject.
    """
    if cfg.verbose:
        print(f"\n{'─'*50}\nSubject {subject_id} | CV mode")

    feats, ae_model = extract_features(windows, cfg, ae_model)
    in_features     = feats.shape[-1]

    if cfg.verbose:
        print(f"  Windows: {windows.shape} | Node features: {feats.shape} "
              f"| in_features={in_features}")

    graphs = build_graphs_for_subject(
        windows=windows,
        node_features=feats,
        labels=labels,
        fs=cfg.fs,
        adj_method=cfg.adj_method,
        threshold=cfg.adj_threshold,
        keep_top_k_percent=cfg.keep_top_k_percent,
    )

    model_kwargs = build_model_kwargs(cfg, in_features)

    metrics = cross_validate_subject(
        graphs=graphs,
        labels=labels,
        model_kwargs=model_kwargs,
        n_classes=n_classes,
        n_splits=cfg.n_cv_folds,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=cfg.device,
        use_class_weights=cfg.use_class_weights,
        verbose=cfg.verbose,
    )

    metrics['subject'] = subject_id
    return metrics, ae_model


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject train / test pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_subject_train_test(
    subject_id:    str,
    train_windows: np.ndarray,   # (N_train, C, T)
    train_labels:  np.ndarray,
    test_windows:  np.ndarray,   # (N_test, C, T)
    test_labels:   np.ndarray,
    cfg:           GNNConfig,
    n_classes:     int,
    ae_model=None,
) -> tuple[dict, object]:
    """
    Train on train split, evaluate on test split.
    AE is trained on train_windows only (no leakage).
    Returns (metrics_dict, trained_model).
    """
    if cfg.verbose:
        print(f"\n{'─'*50}\nSubject {subject_id} | Train→Test mode")

    # Feature extraction — fit AE on train only
    train_feats, ae_model = extract_features(train_windows, cfg, ae_model)
    test_feats,  _        = extract_features(test_windows, cfg, ae_model)

    in_features = train_feats.shape[-1]

    # Build graph lists
    train_graphs = build_graphs_for_subject(
        windows=train_windows, node_features=train_feats,
        labels=train_labels,   fs=cfg.fs,
        adj_method=cfg.adj_method, threshold=cfg.adj_threshold,
        keep_top_k_percent=cfg.keep_top_k_percent,
    )
    test_graphs = build_graphs_for_subject(
        windows=test_windows, node_features=test_feats,
        labels=test_labels,   fs=cfg.fs,
        adj_method=cfg.adj_method, threshold=cfg.adj_threshold,
        keep_top_k_percent=cfg.keep_top_k_percent,
    )

    model_kwargs = build_model_kwargs(cfg, in_features)

    model, test_metrics = train_and_test(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        train_labels=train_labels,
        model_kwargs=model_kwargs,
        n_classes=n_classes,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=cfg.device,
        use_class_weights=cfg.use_class_weights,
        verbose=cfg.verbose,
    )

    test_metrics['subject'] = subject_id
    return test_metrics, model


# ─────────────────────────────────────────────────────────────────────────────
# Cross-subject (all subjects at once)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_subjects(
    total_data: dict,    # {subject_id: {'data': (N,C,T), 'labels': (N,)}}
    test_data:  dict,    # {subject_id: {'data': (N,C,T), 'labels': (N,)}}
    cfg:        GNNConfig,
    n_classes:  int,
    mode:       Literal['cv', 'train_test'] = 'train_test',
) -> dict:
    """
    Runs the pipeline for every subject and collects results.

    Returns:
        {
          subject_id: metrics_dict,
          ...
          '__summary__': {metric_mean, metric_std across subjects}
        }
    """
    all_results = {}

    for subj, subj_data in total_data.items():
        windows = subj_data['data']    # (N, C, T)
        labels  = subj_data['labels']  # (N,)

        if mode == 'cv':
            metrics, _ = run_subject_cv(
                subject_id=subj,
                windows=windows,
                labels=labels,
                cfg=cfg,
                n_classes=n_classes,
            )
        elif mode == 'train_test':
            if subj not in test_data:
                print(f"  ⚠️  No test data for {subj}, skipping.")
                continue
            test_windows = test_data[subj]['data']
            test_labels  = test_data[subj]['labels']
            metrics, _ = run_subject_train_test(
                subject_id=subj,
                train_windows=windows,
                train_labels=labels,
                test_windows=test_windows,
                test_labels=test_labels,
                cfg=cfg,
                n_classes=n_classes,
            )
        else:
            raise ValueError(f"mode must be 'cv' or 'train_test', got {mode!r}")

        all_results[subj] = metrics

    # Summary across subjects
    key_map = {
        'cv':         ['accuracy_mean', 'balanced_accuracy_mean', 'kappa_mean', 'f1_macro_mean'],
        'train_test': ['accuracy', 'balanced_accuracy', 'kappa', 'f1_macro'],
    }
    summary = {}
    for k in key_map[mode]:
        vals = [v[k] for v in all_results.values() if k in v]
        if vals:
            summary[f'{k}_across_subjects_mean'] = float(np.mean(vals))
            summary[f'{k}_across_subjects_std']  = float(np.std(vals))

    all_results['__summary__'] = summary
    return all_results
