import torch
import torch.nn as nn
import torch.nn.functional as F





class EEG_CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7))
        self.pool1 = nn.MaxPool2d((2, 4))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7))
        self.pool2 = nn.MaxPool2d((2, 4))
        self.dropout = nn.Dropout(0.5)

        # Adaptive pooling gives consistent output size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B, 1, 128, 2001)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.global_pool(x)      # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 32)
        x = self.dropout(x)
        return self.fc(x)
