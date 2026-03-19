import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from sklearn.metrics import classification_report

from utils.data import prepare_dataloaders
from utils.train_loop import train_model
from models.eeg_cnn import EEG_CNN
from models.lstm import EEG_LSTM
#from models.eeg_lstm import EEG_LSTM


# ---------- Reproducibility ----------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="EEG Classification Training")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "lstm"],
                        help="Model type to train: cnn or lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load EEG data ---
    # Replace with your real EEG arrays
    # X_imag, y_imag = np.load("X.npy"), np.load("y.npy")
  #  X_imag, y_imag = np.load("/Users/kavinfidel/Documents/Fidel/CNS_Lab/VM_EEG/processed_ica/X_obs.npy") , np.load("/Users/kavinfidel/Documents/Fidel/CNS_Lab/VM_EEG/processed_ica/y_obs.npy")
    X_imag, y_imag = np.load("/Users/kavinfidel/Documents/Fidel/CNS_Lab/VM_EEG/processed_ica/chunked_numpy/X_chunked_img.npy") , np.load("/Users/kavinfidel/Documents/Fidel/CNS_Lab/VM_EEG/processed_ica/chunked_numpy/y_chunked_img.npy")

    train_dl, val_dl = prepare_dataloaders(X_imag, y_imag, batch_size=args.batch_size)

    n_classes = len(np.unique(y_imag))

    # --- Select model ---
    if args.model == "cnn":
        model = EEG_CNN(n_classes).to(device)
    elif args.model == "lstm":
         model = EEG_LSTM(n_classes).to(device)
    else:
        raise ValueError("Unknown model type")

    print(f"🧠 Using model: {args.model.upper()}")
    print(model)

    # --- Training setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
   # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # --- Train ---
    train_model(model, train_dl, val_dl, criterion, optimizer, None, device,
                num_epochs=args.epochs, save_path=args.save_path)

    # --- Evaluate best model ---
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n🧩 Classification Report:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()

# use like: python main.py --model cnn --epochs 40 --batch_size 32
