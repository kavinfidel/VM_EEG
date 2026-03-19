import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

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

def gnn_dataloader(X,y,batch_size = )

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
