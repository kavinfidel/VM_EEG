import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, dataloader, device):
    """Evaluates a trained model on a dataset and prints metrics."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n🧩 Classification Report:")
    print(classification_report(y_true, y_pred))
    print("\n🔢 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return y_true, y_pred
