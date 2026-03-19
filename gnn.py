import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1.  PLV CONNECTIVITY  →  ADJACENCY MATRIX
# ─────────────────────────────────────────────

def compute_plv(segment: np.ndarray) -> np.ndarray:
    """
    Compute Phase Locking Value matrix for one segment.
    
    Args:
        segment: (n_channels, n_samples)  float32/64
    Returns:
        plv_matrix: (n_channels, n_channels)  values in [0, 1]
    """
    n_ch = segment.shape[0]
    # Analytic signal via Hilbert → instantaneous phase
    from scipy.signal import hilbert
    phases = np.angle(hilbert(segment, axis=1))   # (n_ch, n_samples)
    
    plv = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            delta_phi = phases[i] - phases[j]
            plv_val = np.abs(np.mean(np.exp(1j * delta_phi)))
            plv[i, j] = plv_val
            plv[j, i] = plv_val
    np.fill_diagonal(plv, 1.0)
    return plv


def plv_to_edge_index(plv_matrix: np.ndarray, 
                       threshold: float = 0.3):
    """
    Threshold PLV matrix → sparse edge_index + edge_attr.
    
    Args:
        plv_matrix: (n_channels, n_channels)
        threshold:  keep edges where PLV > threshold
    Returns:
        edge_index: (2, n_edges)  long tensor
        edge_attr:  (n_edges, 1)  float tensor  (PLV weights)
    """
    rows, cols = np.where(plv_matrix > threshold)
    # Remove self-loops (diagonal already handled by fill_diagonal but just in case)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_attr  = torch.tensor(plv_matrix[rows, cols], dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


# ─────────────────────────────────────────────
# 2.  NODE FEATURES  FROM  TIME WINDOWS
# ─────────────────────────────────────────────

def extract_node_features(segment: np.ndarray, 
                           sfreq: float = 500.0,
                           window_size: float = 0.6) -> np.ndarray:
    """
    For each channel (node), compute band-power features over 0.6s windows
    then concatenate as the node feature vector.
    
    Feature per window: [mean_power, alpha_power, beta_power, std]
    Final node feature: concatenation across all windows in the 4s segment.
    
    Args:
        segment:     (n_channels, n_samples)
        sfreq:       sampling frequency
        window_size: in seconds
    Returns:
        node_features: (n_channels, n_features)
    """
    from scipy.signal import welch
    
    n_ch, n_samples = segment.shape
    win_samples = int(window_size * sfreq)
    n_windows = n_samples // win_samples
    
    all_features = []
    
    for w in range(n_windows):
        s = w * win_samples
        e = s + win_samples
        chunk = segment[:, s:e]                      # (n_ch, win_samples)
        
        # Welch PSD per channel for this window
        freqs, psd = welch(chunk, fs=sfreq, nperseg=min(win_samples, 128), axis=1)
        
        alpha_mask = (freqs >= 8)  & (freqs <= 13)
        beta_mask  = (freqs >= 14) & (freqs <= 26)
        
        alpha_power = psd[:, alpha_mask].mean(axis=1)   # (n_ch,)
        beta_power  = psd[:, beta_mask].mean(axis=1)    # (n_ch,)
        mean_power  = psd.mean(axis=1)                  # (n_ch,)
        std_amp     = chunk.std(axis=1)                 # (n_ch,)
        
        # Stack → (n_ch, 4)
        window_feat = np.stack([mean_power, alpha_power, beta_power, std_amp], axis=1)
        all_features.append(window_feat)
    
    # Concatenate across windows → (n_ch, 4 * n_windows)
    node_features = np.concatenate(all_features, axis=1).astype(np.float32)
    return node_features


# ─────────────────────────────────────────────
# 3.  DATASET BUILDER
# ─────────────────────────────────────────────

def build_graph_dataset(pipeline, 
                         plv_threshold: float = 0.3,
                         label_dict: dict = None) -> list:
    """
    Converts preprocessed pipeline output into a list of PyG Data objects.
    One graph per 4s OS→OE segment.
    
    Args:
        pipeline:      fitted preprocessing_pipeline instance
        plv_threshold: edge inclusion threshold
        label_dict:    e.g. {'OSBA': 0, 'OSBY': 1, ...}
    Returns:
        graph_list: list of torch_geometric.data.Data
    """
    classes = ['BA', 'BY', 'DO', 'MO', 'SI']
    
    if label_dict is None:
        label_dict = {f'OS{cls}': i for i, cls in enumerate(classes)}
    
    graph_list = []
    
    for cls in classes:
        starts = [ann['onset'] for ann in pipeline.annotations 
                  if ann['description'] == f'OS{cls}']
        ends   = [ann['onset'] for ann in pipeline.annotations 
                  if ann['description'] == f'OE{cls}']
        
        for start, end in zip(starts, ends):
            segment = pipeline.raw.copy().crop(tmin=start, tmax=end)
            data_arr = segment.get_data(picks='eeg').astype(np.float32)
            # data_arr: (n_channels, n_samples)
            
            # --- Adjacency: PLV over full 4s segment ---
            plv_mat = compute_plv(data_arr)
            edge_index, edge_attr = plv_to_edge_index(plv_mat, threshold=plv_threshold)
            
            # --- Node features: band power over 0.6s windows ---
            node_feat = extract_node_features(data_arr, 
                                               sfreq=pipeline.fs,
                                               window_size=pipeline.time_window)
            # node_feat: (n_channels, n_features)
            
            x = torch.tensor(node_feat, dtype=torch.float32)
            y = torch.tensor([label_dict[f'OS{cls}']], dtype=torch.long)
            
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graph_list.append(graph)
    
    print(f"  Built {len(graph_list)} graphs | "
          f"Nodes: {graph_list[0].num_nodes} | "
          f"Node feature dim: {graph_list[0].x.shape[1]} | "
          f"Avg edges: {np.mean([g.num_edges for g in graph_list]):.0f}")
    
    return graph_list


# ─────────────────────────────────────────────
# 4.  GNN MODEL  (GAT + residual + dropout)
# ─────────────────────────────────────────────

class EEG_GAT(nn.Module):
    """
    Graph Attention Network for EEG visual imagery classification.
    
    Architecture:
        GATConv(in  → 64, heads=4)  →  residual proj
        GATConv(256 → 64, heads=4)  →  residual
        GATConv(256 → 64, heads=1)
        GlobalMeanPool
        MLP(64 → 128 → n_classes)
    """
    def __init__(self, in_channels: int, n_classes: int = 5,
                 hidden: int = 64, heads: int = 4, dropout: float = 0.4):
        super().__init__()
        
        self.dropout = dropout
        
        # GAT layers
        self.conv1 = GATConv(in_channels, hidden, heads=heads, 
                              edge_dim=1, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads,
                              edge_dim=1, dropout=dropout)
        self.conv3 = GATConv(hidden * heads, hidden, heads=1,
                              edge_dim=1, dropout=dropout)
        
        # Residual projections (to match dimensions after concatenation)
        self.res1 = nn.Linear(in_channels,    hidden * heads)
        self.res2 = nn.Linear(hidden * heads, hidden * heads)
        self.res3 = nn.Linear(hidden * heads, hidden)
        
        # Batch norms
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.bn3 = nn.BatchNorm1d(hidden)
        
        # Classifier MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.elu(h + self.res1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2
        h2 = self.conv2(h, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.elu(h2 + self.res2(h))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        # Layer 3
        h3 = self.conv3(h2, edge_index, edge_attr)
        h3 = self.bn3(h3)
        h3 = F.elu(h3 + self.res3(h2))
        
        # Global pooling: aggregate all node embeddings → graph embedding
        graph_emb = global_mean_pool(h3, batch)   # (batch_size, hidden)
        
        return self.mlp(graph_emb)


# ─────────────────────────────────────────────
# 5.  TRAINING  LOOP
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        correct    += (out.argmax(dim=1) == batch.y).sum().item()
        total      += batch.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        correct    += (out.argmax(dim=1) == batch.y).sum().item()
        total      += batch.num_graphs
    return total_loss / total, correct / total


def train_gnn(graph_list: list,
              n_classes:  int   = 5,
              epochs:     int   = 100,
              lr:         float = 1e-3,
              batch_size: int   = 8,
              n_folds:    int   = 5,
              device_str: str   = 'auto'):
    """
    Stratified K-Fold cross-validation training loop.
    
    Args:
        graph_list: output of build_graph_dataset()
        n_classes:  number of output classes
        epochs:     training epochs per fold
        lr:         learning rate
        batch_size: graphs per batch
        n_folds:    cross-validation folds
        device_str: 'auto', 'cpu', or 'cuda'
    """
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'auto' else device_str
    )
    print(f"\n Using device: {device}")
    
    labels  = np.array([g.y.item() for g in graph_list])
    in_dim  = graph_list[0].x.shape[1]
    
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(graph_list, labels)):
        print(f"\n{'─'*40}")
        print(f"  Fold {fold+1}/{n_folds}")
        
        train_graphs = [graph_list[i] for i in train_idx]
        val_graphs   = [graph_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)
        
        model     = EEG_GAT(in_channels=in_dim, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc = eval_epoch(model,  val_loader,   criterion, device)
            scheduler.step()
            
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), f'best_fold{fold+1}.pt')
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | "
                      f"Train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                      f"Val loss {va_loss:.3f} acc {va_acc:.3f}")
        
        fold_accs.append(best_val_acc)
        print(f"  ✅ Best val acc fold {fold+1}: {best_val_acc:.4f}")
    
    print(f"\n{'═'*40}")
    print(f"  Mean val acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    return fold_accs


# ─────────────────────────────────────────────
# 6.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # --- 1. Run your pipeline ---
    bad_chs = ['E17', 'E38', 'E94', 'E113', 'E119', 'E121', 'E125',
               'E128', 'E73', 'E81', 'E88', 'E43', 'E44', 'E120', 'E114']
    active_chs = []  # fill in if needed

    pipeline = preprocessing_pipeline(
        'your_file.raw',
        active_chs, bad_chs,
        l_freq=8.0, h_freq=26.0,
        apply_ica=True, remove_muscle=True
    )

    # --- 2. Build graph dataset ---
    label_dict = {f'OS{cls}': i for i, cls in enumerate(['BA', 'BY', 'DO', 'MO', 'SI'])}
    graphs = build_graph_dataset(pipeline, plv_threshold=0.3, label_dict=label_dict)

    # --- 3. Train ---
    fold_accuracies = train_gnn(graphs, n_classes=5, epochs=100, lr=1e-3, batch_size=8)