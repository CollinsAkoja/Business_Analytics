import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
from pathlib import Path

from src.pytorch_anomaly.dataset import AnomalyDataset
from src.pytorch_anomaly.model import FFN

def precision_at_k(y_true, scores, k=0.01):
    n = len(scores)
    topn = max(1, int(n * k))
    idx = np.argsort(-scores)[:topn]
    return y_true[idx].sum() / topn

def train(args):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    ds = AnomalyDataset(args.csv)
    n = len(ds)
    valn = int(0.1*n)
    trainn = n - valn
    train_ds, val_ds = random_split(ds, [trainn, valn])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = ds.X.shape[1]
    model = FFN(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device=="cuda" else None

    best_auc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(X_batch)
                    loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X_batch)
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * X_batch.size(0)
        avg_loss = total_loss / trainn

        model.eval()
        y_true = []
        scores = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                scores.extend(probs)
                y_true.extend(y_batch.numpy())
        y_true = np.array(y_true)
        scores = np.array(scores)
        auc = roc_auc_score(y_true, scores) if len(np.unique(y_true))>1 else 0.0
        p_at_1 = precision_at_k(y_true, scores, k=0.01)
        print(f"Epoch {epoch}, loss={avg_loss:.4f}, val_auc={auc:.4f}, prec@1%={p_at_1:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save({'model_state_dict': model.state_dict()}, args.checkpoint)
    print('Best val AUC', best_auc)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/supermarket_transactions.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="pytorch_anom_best.pth")
    args = parser.parse_args()
    Path("data").mkdir(parents=True, exist_ok=True)
    train(args)
