"""
graph.py
========
Builds adjacency matrices and converts (features, adjacency) pairs into
PyTorch Geometric `Data` objects.

Two connectivity matrices:

  1. PLV  (Phase Locking Value) — recommended for EEG functional connectivity.
     PLV_{ij} = |mean(exp(i * (φ_i(t) - φ_j(t))))| over time.
     Computed per window per frequency band, then averaged across bands.
     Range [0, 1]; higher = stronger phase synchrony.

  2. Pearson — linear amplitude correlation.
     Fast and simple; more susceptible to volume conduction.

Both return a (N_windows, C, C) adjacency array.

Thresholding:
  - Absolute threshold: keep edges where weight > τ.
  - Sparsity target: keep top-k% edges per node (proportional pruning).
  - Self-loops are always removed.

Node features + adjacency → PyG Data
  Each window becomes one graph:
    x    : (C, F)          node feature matrix
    edge_index : (2, E)    COO sparse edge list
    edge_attr  : (E, 1)    edge weights (connectivity strength)
    y    : scalar label
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from typing import Literal, Optional

import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BANDS = {
    'theta': (4,  8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}


def _bandpass(signal: np.ndarray, fs: float,
              low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. signal: (C, T)"""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PLV adjacency
# ─────────────────────────────────────────────────────────────────────────────

def _plv_single_band(data: np.ndarray) -> np.ndarray:
    """
    Compute PLV matrix for one (C, T) window in one frequency band.
    Returns (C, C) PLV matrix.
    """
    analytic = hilbert(data, axis=-1)            # (C, T) complex
    phase    = np.angle(analytic)                # (C, T) instantaneous phase

    # PLV_{ij} = |mean_t( exp(i*(phi_i - phi_j)) )|
    # Efficient vectorised computation:
    exp_phi = np.exp(1j * phase)                 # (C, T)
    # outer product of mean over time
    plv = np.abs(exp_phi @ exp_phi.conj().T) / data.shape[-1]
    np.fill_diagonal(plv, 0.0)
    return plv.astype(np.float32)


def compute_plv_adjacency(
    windows: np.ndarray,    # (N, C, T)
    fs: float,
    bands: Optional[dict] = None,
) -> np.ndarray:
    """
    Returns PLV adjacency matrices of shape (N, C, C).
    Averaged across the frequency bands defined in `bands`.
    """
    if bands is None:
        bands = BANDS

    N, C, T = windows.shape
    adj = np.zeros((N, C, C), dtype=np.float32)

    for band_name, (low, high) in bands.items():
        for n in range(N):
            filtered = _bandpass(windows[n], fs, low, high)
            adj[n]  += _plv_single_band(filtered)

    adj /= len(bands)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Pearson adjacency
# ─────────────────────────────────────────────────────────────────────────────

def compute_pearson_adjacency(
    windows: np.ndarray,   # (N, C, T)
) -> np.ndarray:
    """
    Returns absolute Pearson correlation adjacency (N, C, C).
    We take absolute value so inhibitory (negative) correlations still
    contribute to connectivity strength.
    """
    N, C, T = windows.shape
    adj = np.zeros((N, C, C), dtype=np.float32)

    for n in range(N):
        x   = windows[n]                          # (C, T)
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-8
        xn  = (x - mu) / std                     # normalised
        corr = (xn @ xn.T) / T                   # (C, C)
        corr = np.abs(corr).astype(np.float32)
        np.fill_diagonal(corr, 0.0)
        adj[n] = corr

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency(
    windows: np.ndarray,
    fs: float,
    method: Literal['plv', 'pearson'] = 'plv',
    plv_bands: Optional[dict] = None,
) -> np.ndarray:
    """
    Unified adjacency builder.
    Returns (N, C, C) float32 connectivity matrices.
    """
    if method == 'plv':
        return compute_plv_adjacency(windows, fs, bands=plv_bands)
    elif method == 'pearson':
        return compute_pearson_adjacency(windows)
    else:
        raise ValueError(f"Unknown adjacency method: {method!r}. Choose 'plv' or 'pearson'.")


# ─────────────────────────────────────────────────────────────────────────────
# Thresholding / sparsification
# ─────────────────────────────────────────────────────────────────────────────

def threshold_adjacency(
    adj: np.ndarray,                   # (N, C, C)
    threshold: Optional[float] = None,
    keep_top_k_percent: Optional[float] = None,
) -> np.ndarray:
    """
    Sparsify adjacency matrices.

    Args:
        threshold:          Hard threshold τ; zero out entries ≤ τ.
        keep_top_k_percent: Per-window retain only top k% of edge weights.
                            Applied after hard threshold if both specified.

    Returns (N, C, C) sparse adjacency (zeros for removed edges).
    """
    adj = adj.copy()

    if threshold is not None:
        adj[adj <= threshold] = 0.0

    if keep_top_k_percent is not None:
        assert 0 < keep_top_k_percent <= 100
        N, C, _ = adj.shape
        for n in range(N):
            flat = adj[n].flatten()
            nonzero = flat[flat > 0]
            if nonzero.size == 0:
                continue
            k_val = np.percentile(nonzero, 100.0 - keep_top_k_percent)
            mask  = adj[n] >= k_val
            adj[n] = adj[n] * mask

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Convert to PyTorch Geometric Data objects
# ─────────────────────────────────────────────────────────────────────────────

def to_pyg_data(
    node_features: np.ndarray,   # (N, C, F)
    adjacency: np.ndarray,       # (N, C, C)  — already thresholded
    labels: np.ndarray,          # (N,)  integer class labels
) -> list[Data]:
    """
    Converts arrays to a list of PyG Data objects, one per window.

    Each Data object:
        x          : FloatTensor  (C, F)
        edge_index : LongTensor   (2, E)
        edge_attr  : FloatTensor  (E, 1)
        y          : LongTensor   scalar
    """
    assert node_features.shape[0] == adjacency.shape[0] == labels.shape[0]
    N = node_features.shape[0]
    graphs = []

    for n in range(N):
        x   = torch.from_numpy(node_features[n]).float()   # (C, F)
        adj = adjacency[n]                                  # (C, C)

        # COO edge list from upper triangle (graph is undirected)
        src, dst = np.where(adj > 0)
        weights  = adj[src, dst]

        # Make undirected: include both (i→j) and (j→i)
        edge_index = torch.tensor(
            np.stack([
                np.concatenate([src, dst]),
                np.concatenate([dst, src]),
            ], axis=0),
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            np.concatenate([weights, weights]),
            dtype=torch.float,
        ).unsqueeze(1)   # (E, 1)

        y = torch.tensor(int(labels[n]), dtype=torch.long)

        graphs.append(Data(x=x, edge_index=edge_index,
                           edge_attr=edge_attr, y=y))

    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full pipeline for one subject's data
# ─────────────────────────────────────────────────────────────────────────────

def build_graphs_for_subject(
    windows: np.ndarray,           # (N, C, T)
    node_features: np.ndarray,     # (N, C, F)
    labels: np.ndarray,            # (N,)
    fs: float,
    adj_method: Literal['plv', 'pearson'] = 'plv',
    threshold: Optional[float] = 0.3,
    keep_top_k_percent: Optional[float] = None,
    plv_bands: Optional[dict] = None,
) -> list[Data]:
    """
    End-to-end: windows → adjacency → threshold → PyG graph list.
    """
    adj = build_adjacency(windows, fs, method=adj_method, plv_bands=plv_bands)
    adj = threshold_adjacency(adj, threshold=threshold,
                              keep_top_k_percent=keep_top_k_percent)
    graphs = to_pyg_data(node_features, adj, labels)
    return graphs
