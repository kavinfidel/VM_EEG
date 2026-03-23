import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from torch_geometric.loader import DataLoader
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from models.RiemannianGAT import RiemannianGAT
from torch_geometric.loader import DataLoader

def prepare_dataloaders(X, y, batch_size=16, split_ratio=0.8):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import numpy as np

    # Normalize per channel
    X_mean = X.mean(axis=(0, 2), keepdims=True)
    X_std = X.std(axis=(0, 2), keepdims=True)
    X_norm = (X - X_mean) / (X_std + 1e-6)

    # Reorder to (B, 1, C, T)
    X_torch = torch.tensor(X_norm, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)
    y_torch = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_torch, y_torch)
    n_total = len(dataset)
    n_train = int(split_ratio * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    return train_dl, val_dl

def convert_to_graph_list(X_data, y_labels, threshold=0.3):
    """
    Bridges the gap: Numpy (Trials, Chs, Time) -> List of PyG Data objects
    """
    graph_list = []
    
    # Use OAS estimator for stable covariance with 118 channels
    cov_est = Covariances(estimator='oas')
    covariances = cov_est.fit_transform(X_data) # Shape: (Trials, 118, 118)

    for i in range(len(covariances)):
        # x: Use the covariance rows as node features (118 nodes, 118 features each)
        # This captures how each node relates to every other node geometrically.
        x = torch.tensor(covariances[i], dtype=torch.float32)
        
        # edge_index: Define connectivity based on covariance strength
        adj = np.abs(covariances[i])
        rows, cols = np.where(adj > threshold)
        
        # Remove self-loops
        mask = rows != cols
        edge_index = torch.tensor(np.stack([rows[mask], cols[mask]], axis=0), dtype=torch.long)
        
        # edge_attr: The actual covariance value as weight
        edge_attr = torch.tensor(adj[rows[mask], cols[mask]], dtype=torch.float32).unsqueeze(1)
        
        y = torch.tensor([y_labels[i]], dtype=torch.long)
        
        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        
    return graph_list

# def prepare_dataloaders(X, y, batch_size=16, split_ratio=0.8): # global
#     import torch
#     from torch.utils.data import TensorDataset, DataLoader, random_split

#     # 🔹 Global Min–Max normalization to [-1, 1]
#     X_min = X.min()
#     X_max = X.max()
#     X_norm = 2 * (X - X_min) / (X_max - X_min) - 1

#     # Convert to torch tensors
#     X_torch = torch.tensor(X_norm, dtype=torch.float32)
#     y_torch = torch.tensor(y, dtype=torch.long)

#     # Split dataset
#     dataset = TensorDataset(X_torch, y_torch)
#     n_total = len(dataset)
#     n_train = int(split_ratio * n_total)
#     n_val = n_total - n_train

#     train_ds, val_ds = random_split(dataset, [n_train, n_val])
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_dl = DataLoader(val_ds, batch_size=batch_size)

#     return train_dl, val_dl
