import torch
import argparse
import numpy as np
from models.eeg_cnn import EEG_CNN
from models.eeg_lstm import EEG_LSTM
from utils.data import prepare_dataloaders
from utils.test_loop import evaluate_model
from models.RiemannianGAT import RiemannianGAT

def main():
    parser = argparse.ArgumentParser(description="EEG Model Testing")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "lstm"],
                        help="Model type to load")
    parser.add_argument("--weights", type=str, default="best_model.pt",
                        help="Path to the trained model weights")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # --- Load EEG data ---
    # Replace with your test EEG arrays
    # X_test, y_test = np.load("X_test.npy"), np.load("y_test.npy")
    # Example placeholder:
    X_test, y_test = np.random.randn(100, 152, 2001), np.random.randint(0, 4, size=(100,))

    _, test_dl = prepare_dataloaders(X_test, y_test, batch_size=args.batch_size, split_ratio=0.0)
    # split_ratio=0.0 → we’ll just get one dataloader (no split)

    n_classes = len(np.unique(y_test))

    # --- Initialize model ---
    if args.model == "cnn":
        model = EEG_CNN(n_classes).to(device)
    elif args.model == "lstm":
        model = EEG_LSTM(n_classes).to(device)
    elif arg.model =='RGAT':
        model = RiemannianGAT(n_classes).to(device)
    else:
        raise ValueError("Unknown model type")

    # --- Load trained weights ---
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"✅ Loaded model weights from: {args.weights}")

    # --- Evaluate ---
    evaluate_model(model, test_dl, device)


if __name__ == "__main__":
    main()

# use lie: python test.py --model cnn --weights best_model.pt
