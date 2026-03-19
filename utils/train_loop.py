import torch
from tqdm import tqdm

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
