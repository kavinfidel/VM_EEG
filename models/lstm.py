import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_LSTM(nn.Module):
    def __init__(self, n_classes, input_size=100, hidden_size=128, num_layers=1):
        super().__init__()
        
        # LSTM expects (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,      # 100 channels
            hidden_size=hidden_size,     # 128 hidden units
            num_layers=num_layers,       # 2 stacked LSTM layers
            batch_first=True,            # Input shape: (batch, seq, features)
            dropout=0.4,                 # Dropout between LSTM layers
            bidirectional=True           # Process sequence forward & backward
        )
        
        # After bidirectional LSTM, size is hidden_size * 2
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape: (batch, 667, 100) ✅ Perfect for LSTM!
        
        # LSTM output: (batch, seq_len, hidden_size * 2)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last time step output
        x = lstm_out[:, -1, :]  # (batch, hidden_size * 2)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x