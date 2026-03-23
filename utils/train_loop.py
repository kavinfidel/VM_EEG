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


def train_model(model, train_dl, val_dl, criterion, optimizer, scheduler, device, num_epochs=30, save_path="best_model.pt"):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ----- TRAIN -----
        model.train()
        running_loss, correct, total = 0.0, 0,0
        train_bar = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)

        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_train_loss = running_loss / total
        train_acc = 100 * correct / total

        # ----- VALIDATION -----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_val, y_val in val_dl:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
       # scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
        )
             # f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f" Best model saved to: {save_path}")


def train_riemannian_gnn(train_dict, test_dict, subject, epochs=500, batch_size=32):
    # Setup Device
    device = torch.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Training Subject: {subject} on {device} ---")

    # 1. Data Prep & Validation Split
    # Since you have small data, we split the train_dict further to get a validation set
    raw_data = train_dict[subject]['data']
    raw_labels = train_dict[subject]['labels']
    
    # Stratified split ensures class balance in tiny datasets
    X_train, X_val, y_train, y_val = train_test_split(
        raw_data, raw_labels, test_size=0.2, stratify=raw_labels, random_state=42
    )

    train_graphs = convert_to_graph_list(X_train, y_train)
    val_graphs = convert_to_graph_list(X_val, y_val)
    test_graphs = convert_to_graph_list(test_dict[subject]['data'], test_dict[subject]['labels'])

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model
    model = RiemannianGAT(in_channels=100, hidden_channels=64, out_channels=3).to(device)
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