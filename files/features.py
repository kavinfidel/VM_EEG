"""
features.py
===========
Two feature extraction strategies for EEG → GNN nodes.

Strategy A: Autoencoder latent representation
    - A 1-D convolutional autoencoder is trained (per-subject or globally)
      on the raw windowed EEG segments.
    - The encoder output (latent vector) becomes the node feature vector.
    - Each *channel* is treated as a node, so the AE operates on a single
      channel's time-series and produces a latent embedding of length
      `latent_dim`.  This keeps the GNN graph structure clean:
      N_channels nodes, each with a `latent_dim`-dimensional feature.

Strategy B: Classical hand-crafted features
    - Differential Entropy (DE) in 5 standard EEG bands.
    - Band-power asymmetry (RASM) between symmetric channel pairs.
    - Hjorth parameters (Activity, Mobility, Complexity) per channel.
    - Total feature vector per channel: 5 (DE) + 5 (RASM, zero-padded for
      unpaired channels) + 3 (Hjorth) = 13 features per node.

Both strategies return a tensor of shape (N_windows, N_channels, F).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import welch
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Frequency band definitions
# ─────────────────────────────────────────────────────────────────────────────

BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy B – Classical features
# ─────────────────────────────────────────────────────────────────────────────

def differential_entropy(psd: np.ndarray, freqs: np.ndarray,
                          band: tuple[float, float]) -> float:
    """
    DE = 0.5 * log(2πe * σ²)  ≈  0.5 * log(2πe * band_power)
    We approximate band power via the trapezoidal PSD integral.
    """
    idx = np.where((freqs >= band[0]) & (freqs < band[1]))[0]
    if idx.size == 0:
        return 0.0
    power = np.trapz(psd[idx], freqs[idx])
    power = max(power, 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * power)


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    """
    Activity   = var(x)
    Mobility   = sqrt(var(dx/dt) / var(x))
    Complexity = Mobility(dx/dt) / Mobility(x)
    """
    activity = np.var(signal)
    if activity < 1e-12:
        return 0.0, 0.0, 0.0
    d1 = np.diff(signal)
    var_d1 = np.var(d1)
    mobility = np.sqrt(var_d1 / activity)
    if mobility < 1e-12:
        return activity, 0.0, 0.0
    d2 = np.diff(d1)
    var_d2 = np.var(d2)
    mob_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 1e-12 else 0.0
    complexity = mob_d1 / mobility if mobility > 1e-12 else 0.0
    return activity, mobility, complexity


def compute_classical_features(
    windows: np.ndarray,       # (N_windows, N_ch, T)
    fs: float,
    symmetric_pairs: Optional[list[tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Returns z-score normalised features of shape (N_windows, N_ch, F)
    where F = 5 (DE) + 5 (RASM) + 3 (Hjorth) = 13.

    All 13 features are globally z-scored across the N_windows dimension
    per (channel, feature) so they have comparable scale entering the GNN.
    Hjorth Activity in particular has very large raw values (EEG variance
    units) that would dominate the first GNN layer without normalisation.

    symmetric_pairs: list of (left_idx, right_idx) channel index pairs
        used to compute RASM = DE_left - DE_right per band.
        Channels without a pair get RASM = 0.
    """
    N, C, T = windows.shape
    n_bands  = len(BANDS)
    F        = n_bands + n_bands + 3   # DE + RASM + Hjorth
    features = np.zeros((N, C, F), dtype=np.float32)

    # Pre-build RASM lookup
    rasm_partner = {}
    if symmetric_pairs:
        for left, right in symmetric_pairs:
            rasm_partner[left]  = (right, +1)
            rasm_partner[right] = (left,  -1)

    for n in range(N):
        freqs, psd = welch(windows[n], fs=fs, nperseg=min(T, 256), axis=-1)

        de_matrix = np.zeros((C, n_bands), dtype=np.float32)
        for b_idx, (_, band) in enumerate(BANDS.items()):
            for c in range(C):
                de_matrix[c, b_idx] = differential_entropy(psd[c], freqs, band)

        features[n, :, :n_bands] = de_matrix

        for c in range(C):
            if c in rasm_partner:
                partner, sign = rasm_partner[c]
                features[n, c, n_bands:n_bands * 2] = sign * (
                    de_matrix[c] - de_matrix[partner]
                )

        for c in range(C):
            act, mob, comp = hjorth_parameters(windows[n, c])
            features[n, c, n_bands * 2]     = act
            features[n, c, n_bands * 2 + 1] = mob
            features[n, c, n_bands * 2 + 2] = comp

    # ── Global z-score: normalise each (channel, feature) across N windows ──
    # Shape (N, C, F) → mean/std over axis 0
    feat_mean = features.mean(axis=0, keepdims=True)        # (1, C, F)
    feat_std  = features.std(axis=0, keepdims=True) + 1e-8  # (1, C, F)
    features  = (features - feat_mean) / feat_std

    return features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy A – Convolutional Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class ChannelEncoder(nn.Module):
    """
    1-D convolutional encoder for a single EEG channel time-series.
    Input:  (batch, 1, T)
    Output: (batch, latent_dim)
    """
    def __init__(self, time_steps: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # T/2
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # T/4
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # T/8
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(4),                                # → (64, 4)
            nn.Flatten(),                                           # → 256
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ChannelDecoder(nn.Module):
    """
    Mirrors the encoder for reconstruction.
    Input:  (batch, latent_dim)
    Output: (batch, 1, T)  (approximate, via interpolation)
    """
    def __init__(self, time_steps: int, latent_dim: int = 32):
        super().__init__()
        self.time_steps = time_steps
        self.decode_fc = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 4)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.decode_fc(z)
        x = self.decoder(x)
        # Resize to original T via interpolation
        x = nn.functional.interpolate(x, size=self.time_steps, mode='linear',
                                       align_corners=False)
        return x


class EEGChannelAutoencoder(nn.Module):
    def __init__(self, time_steps: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = ChannelEncoder(time_steps, latent_dim)
        self.decoder = ChannelDecoder(time_steps, latent_dim)

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_autoencoder(
    windows: np.ndarray,     # (N, C, T)
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = True,
) -> EEGChannelAutoencoder:
    """
    Train the channel-level AE on all (window, channel) pairs.
    We flatten (N, C, T) → (N*C, 1, T) so the AE sees individual channels.
    Returns the trained model (encoder can be used for feature extraction).
    """
    # Use the same resolve_device logic as train.py (CUDA > MPS > CPU)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    N, C, T = windows.shape
    # Flatten channels into the batch dimension
    data = windows.reshape(N * C, 1, T).astype(np.float32)

    # Z-score per sample — essential: raw EEG amplitudes vary hugely across
    # channels and subjects. Without this the AE loss is dominated by scale,
    # not shape, and the latents encode amplitude rather than dynamics.
    mu  = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    data = (data - mu) / std

    tensor  = torch.from_numpy(data)
    dataset = TensorDataset(tensor)
    # pin_memory only valid for CUDA
    pin = (device == 'cuda')
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=pin)

    model = EEGChannelAutoencoder(time_steps=T, latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    if verbose:
        print(f"  AE: {N*C} samples ({N} windows × {C} channels) | "
              f"latent_dim={latent_dim} | device={device}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        scheduler.step()
        if verbose and epoch % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  AE Epoch {epoch:3d}/{epochs} | Loss: {avg:.6f}")

    model.eval()
    return model


@torch.no_grad()
def extract_ae_features(
    model: EEGChannelAutoencoder,
    windows: np.ndarray,    # (N, C, T)
    batch_size: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Returns z-score normalised latent features of shape (N, C, latent_dim).

    Normalisation note: raw latents have arbitrary scale per dimension.
    We z-score across the (N*C) sample dimension so every latent dimension
    has zero mean and unit variance before entering the GNN. Without this
    the GNN's first-layer BatchNorm sees inputs that are already near-constant
    per feature, which causes it to collapse and output a single class.
    """
    # Resolve device from the model itself, or override with passed value
    model_device = str(next(model.parameters()).device)
    device = device if device is not None else model_device
    model.eval()

    N, C, T = windows.shape
    data = windows.reshape(N * C, 1, T).astype(np.float32)
    # Apply the same per-sample z-score as during training
    mu   = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True) + 1e-8
    data = (data - mu) / std

    tensor  = torch.from_numpy(data)
    dataset = TensorDataset(tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0)

    latents = []
    for (batch,) in loader:
        batch = batch.to(device)
        _, z  = model(batch)
        latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, axis=0)           # (N*C, latent_dim)
    latents = latents.reshape(N, C, -1).astype(np.float32)  # (N, C, latent_dim)

    # ── Global z-score normalisation across all (N*C) samples per latent dim ──
    # Shape: (N*C, latent_dim) → compute mean/std over axis 0
    flat = latents.reshape(N * C, -1)                   # (N*C, latent_dim)
    feat_mean = flat.mean(axis=0, keepdims=True)        # (1, latent_dim)
    feat_std  = flat.std(axis=0, keepdims=True) + 1e-8  # (1, latent_dim)
    flat_norm = (flat - feat_mean) / feat_std
    return flat_norm.reshape(N, C, -1).astype(np.float32)
