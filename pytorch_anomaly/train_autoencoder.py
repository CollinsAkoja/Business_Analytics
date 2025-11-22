import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path

from src.pytorch_anomaly.dataset import AnomalyDataset
from src.pytorch_anomaly.autoencoder_model import AutoEncoder

def train_ae(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = AnomalyDataset(args.csv)
    n = len(ds)
    valn = int(0.1*n)
    trainn = n - valn
    train_ds, val_ds = random_split(ds, [trainn, valn])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = ds.X.shape[1]
    model = AutoEncoder(input_dim, latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            opt.zero_grad()
            recon, _ = model(X_batch)
            loss = F.mse_loss(recon, X_batch)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * X_batch.size(0)
        avg_loss = total_loss / trainn

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                recon, _ = model(X_batch)
                loss = F.mse_loss(recon, X_batch)
                val_loss += float(loss.item()) * X_batch.size(0)
        val_loss = val_loss / valn
        print(f"Epoch {epoch}, train_mse={avg_loss:.6f}, val_mse={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, args.checkpoint)
    print("Best val MSE", best_val_loss)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/supermarket_transactions.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default="pytorch_ae_best.pth")
    args = parser.parse_args()
    Path("data").mkdir(parents=True, exist_ok=True)
    train_ae(args)
