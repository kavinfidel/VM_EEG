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
        # 1. Convert the main matrix to a tensor immediately
        cov_tensor = torch.from_numpy(covariances[i]).float()
        
        # 2. Use the tensor for node features
        x = cov_tensor 
        
        # 3. Use PyTorch operations for the adjacency/edges
        adj = torch.abs(cov_tensor)
        
        # Find indices where adj > threshold (equivalent to np.where)
        edge_index = (adj > threshold).nonzero(as_tuple=False).t()
        
        # 4. Filter self-loops using torch logic
        mask = edge_index != edge_index
        edge_index = edge_index[:, mask]
        
        # 5. Get edge attributes
        edge_attr = adj[edge_index, edge_index].unsqueeze(1)
        
        y = torch.tensor([y_labels[i]], dtype=torch.long)
        
        # Now all components are guaranteed Tensors
        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        
    return graph_list