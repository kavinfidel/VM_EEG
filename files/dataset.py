"""
dataset.py
==========
PyTorch Geometric InMemoryDataset wrapper for the pre-built graph lists.

Usage
-----
    graphs = build_graphs_for_subject(windows, node_feats, labels, fs=250)
    ds = EEGGraphDataset(graphs)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

from torch_geometric.data import InMemoryDataset, Data
from typing import List


class EEGGraphDataset(InMemoryDataset):
    """
    Lightweight in-memory dataset wrapping a list of PyG Data objects.
    No disk caching — everything lives in RAM.
    """

    def __init__(self, graphs: List[Data]):
        super().__init__(root=None, transform=None)
        self.data, self.slices = self.collate(graphs)

    def _download(self):
        pass

    def _process(self):
        pass
