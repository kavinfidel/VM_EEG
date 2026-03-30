import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU, Dropout
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGELite(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGELite, self).__init__()
        
        # Reduced to 2 layers to prevent "over-smoothing" 
        # (where all nodes start looking identical)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Simplified head: Just one linear layer
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Readout: Use Mean Pool only (more stable for small data)
        x = global_mean_pool(x, batch) 
        
        # Final prediction
        return self.classifier(x)