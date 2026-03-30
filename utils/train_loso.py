import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.data import convert_to_graph_list
import numpy as np
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from models.RiemannianGAT import RiemannianGAT
from torch_geometric.loader import DataLoader


def train_loso_riemannian_gnn(train_dict, test_dict, test_sub, epochs=1000, batch_size=8):
    # Setup Device
    device = torch.device("cpu")
    print(f"--- Testing Subject: {test_sub} on {device} ---")

    sub_list = ['S116', 'S118', 'S119', 'S117', 'S2_', 'S1_', 'S115', 'S113', 'S114']
    sub_list.remove(test_sub)
    
    # 1. Data Prep & Validation Split
    # Since you have small data, we split the train_dict further to get a validation set
    x_all = []
    y_all = []

    for subject in sub_list:
        raw_data = train_dict[subject]['data']
        raw_labels = train_dict[subject]['labels']
        x_all.extend(raw_data)
        y_all.extend(raw_labels)
    
    
    # Stratified split ensures class balance in tiny datasets
    X_train, X_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    X_train = np.array(X_train)
    X_val   = np.array(X_val)
    y_train = np.array(y_train)
    y_val   = np.array(y_val)


    train_graphs = convert_to_graph_list(X_train, y_train)
    val_graphs = convert_to_graph_list(X_val, y_val)
    test_graphs = convert_to_graph_list(test_dict[test_sub]['data'], test_dict[test_sub]['labels'])

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model
    model = RiemannianGAT(in_channels=29, hidden_channels=64, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Added weight decay for small data
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    
    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device) # Move data to MPS
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 4. Validation Phase (The "Better" Part)
        model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                total_val_loss += criterion(out, batch.y).item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        acc = correct / len(val_graphs)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}")
            
        # Simple Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_{subject}.pth') # Optional: Save best weights

    return model, test_loader