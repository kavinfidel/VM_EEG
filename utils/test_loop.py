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


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_riemannian_gnn(model, test_loader, device, class_names=['BA', 'DO', 'SI'],weight = 'best_model.pt'):
    #model = torch.load(best_model)

    if weight:
        state_dict = torch.load(weight, map_location=device)
        model.load_state_dict(state_dict)
   
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    print("🧐 Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # --- 1. Metrics ---
    print("\n✅ Evaluation Complete.")
    print(classification_report(all_labels, all_preds, target_names=class_names[:len(np.unique(all_labels))]))

    # --- 2. Confusion Matrix Plot ---
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names[:len(np.unique(all_labels))])
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')
    plt.title("GNN Riemannian Confusion Matrix")
    plt.show()

    return all_labels, all_preds