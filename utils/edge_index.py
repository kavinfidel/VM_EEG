import torch
from torch_geometric.data import Data
from pyriemann.estimation import Covariances

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