"""
models.py
=========
GNN classifiers for EEG motor imagery.

Three backbone architectures (selectable via `arch` parameter):

  • 'gcn'       – Graph Convolutional Network (Kipf & Welling 2017)
                  Baseline. Fast, good for smoothly-connected graphs.

  • 'graphsage' – GraphSAGE (Hamilton et al. 2017)
                  Aggregates neighbourhood via mean/max/lstm.
                  Robust, generalises well across subjects.
                  ★ Default recommended for EEG.

  • 'gat'       – Graph Attention Network (Veličković et al. 2018)
                  Learns attention weights per edge → interpretable.
                  Slightly slower but reveals which channel pairs matter.

All models share the same interface:
    model = EEGGraphNet(in_features, n_classes, arch='graphsage', ...)
    logits = model(data)     # data is a PyG Batch

Readout: global mean + max pooling concatenated → MLP head.
This is more expressive than mean-only and avoids over-smoothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.data import Batch


# ─────────────────────────────────────────────────────────────────────────────
# Shared readout + classifier head
# ─────────────────────────────────────────────────────────────────────────────

class GraphReadoutHead(nn.Module):
    """
    Global pooling (mean || max) followed by a 2-layer MLP.
    Input graph embedding dim = 2 * hidden_dim  (after concat).
    """
    def __init__(self, hidden_dim: int, n_classes: int,
                 dropout: float = 0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_pool = global_mean_pool(x, batch)   # (B, hidden_dim)
        max_pool  = global_max_pool(x, batch)    # (B, hidden_dim)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)
        return self.mlp(graph_emb)


# ─────────────────────────────────────────────────────────────────────────────
# GCN backbone
# ─────────────────────────────────────────────────────────────────────────────

class GCNBackbone(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int,
                 n_layers: int = 3, dropout: float = 0.4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_features] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1],
                                       add_self_loops=True,
                                       normalize=True))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE backbone
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGEBackbone(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int,
                 n_layers: int = 3, dropout: float = 0.4,
                 aggr: str = 'mean'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_features] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GAT backbone (GATv2 — fixes static attention of the original GAT)
# ─────────────────────────────────────────────────────────────────────────────

class GATBackbone(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int,
                 n_layers: int = 3, heads: int = 4,
                 dropout: float = 0.4, edge_dim: int = 1):
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"

        head_dim = hidden_dim // heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = in_features
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            out_heads = 1 if is_last else heads
            out_hdim  = hidden_dim if is_last else head_dim
            self.convs.append(
                GATv2Conv(in_dim, out_hdim, heads=out_heads,
                          dropout=dropout, edge_dim=edge_dim,
                          concat=(not is_last))
            )
            self.norms.append(nn.BatchNorm1d(
                out_hdim * out_heads if not is_last else out_hdim
            ))
            in_dim = out_hdim * out_heads if not is_last else out_hdim

        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Unified model
# ─────────────────────────────────────────────────────────────────────────────

BACKBONES = {
    'gcn':       GCNBackbone,
    'graphsage': GraphSAGEBackbone,
    'gat':       GATBackbone,
}


class EEGGraphNet(nn.Module):
    """
    Unified EEG GNN classifier.

    Args:
        in_features : Number of features per node (F).
        n_classes   : Number of motor imagery classes.
        arch        : 'gcn' | 'graphsage' | 'gat'
        hidden_dim  : Hidden dimension in GNN layers.
        n_layers    : Number of GNN message-passing layers.
        dropout     : Dropout probability.
        **kwargs    : Extra args forwarded to the backbone
                      (e.g. aggr='mean' for SAGE, heads=4 for GAT).
    """
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        arch: str = 'graphsage',
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.4,
        **kwargs,
    ):
        super().__init__()
        assert arch in BACKBONES, \
            f"Unknown arch '{arch}'. Choose from {list(BACKBONES.keys())}."

        self.backbone = BACKBONES[arch](
            in_features=in_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )
        self.head = GraphReadoutHead(
            hidden_dim=self.backbone.out_dim,
            n_classes=n_classes,
            dropout=dropout,
        )

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.backbone(
            data.x,
            data.edge_index,
            edge_attr=getattr(data, 'edge_attr', None),
            batch=data.batch,
        )
        return self.head(x, data.batch)

    def predict_proba(self, data: Batch) -> torch.Tensor:
        return F.softmax(self.forward(data), dim=-1)
