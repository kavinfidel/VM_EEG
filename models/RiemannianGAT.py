import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class RiemannianGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # GATv2 is more robust for high-dimensional sensor data
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1)
        
        self.post_conv = nn.Linear(hidden_channels * heads, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # x: (Nodes, 118 features)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        
        # Global Pooling: Collapse 118 nodes into 1 graph representation
        x = global_max_pool(x, batch) 
        
        x = self.post_conv(x)
        return self.classifier(x)