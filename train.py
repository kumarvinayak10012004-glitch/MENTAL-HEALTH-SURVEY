import os, json, time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from src.preprocess import fit_transform_save, TARGET_COL_DEFAULT, DROP_COLS_DEFAULT
from src.model import MLP

def load_arrays(artifacts_dir: str):
    X_train = np.load(os.path.join(artifacts_dir, "X_train.npy"))
    X_val = np.load(os.path.join(artifacts_dir, "X_val.npy"))
    y_train = np.load(os.path.join(artifacts_dir, "y_train.npy"))
    y_val = np.load(os.path.join(artifacts_dir, "y_val.npy"))
    with open(os.path.join(artifacts_dir, "schema.json")) as f:
        schema = json.load(f)
    return X_train, X_val, y_train, y_val, schema

def train_model(artifacts_dir="artifacts", epochs=10, batch_size=256, lr=1e-3, device=None):
    X_train, X_val, y_train, y_val, schema = load_arrays(artifacts_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim=int(X_train.shape[1]))
    model.to(device)

    # Handle class imbalance with positive class weight
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_f1, best_path = -1.0, os.path.join(artifacts_dir, "model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        # Eval
        model.eval()
        all_probs, all_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_trues.append(yb.numpy())
        probs = np.concatenate(all_probs)
        y_true = np.concatenate(all_trues)
        y_pred = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = float("nan")

        print(f"Epoch {epoch:02d} | loss={epoch_loss/len(train_ds):.4f} | acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f} | auc={auc:.4f}")

    # ---- Fairness / subgroup metrics (if val_meta.csv exists) ----
    val_meta_path = os.path.join(artifacts_dir, "val_meta.csv")
    if os.path.exists(val_meta_path):
        val_meta = pd.read_csv(val_meta_path)
        val_meta = val_meta.reset_index(drop=True)
        # align lengths (just in case)
        n = min(len(val_meta), len(y_true))
        val_meta = val_meta.iloc[:n]
        y_true_sub = y_true[:n]
        y_pred_sub = y_pred[:n]

        def subgroup_report(df, by):
            groups = df[by].fillna("NA").astype(str).unique().tolist()
            rows = []
            for g in groups:
                idx = (df[by].astype(str) == g).values
                if idx.sum() == 0: 
                    continue
                yt = y_true_sub[idx]
                yp = y_pred_sub[idx]
                acc_g = accuracy_score(yt, yp)
                p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
                rows.append([g, idx.sum(), acc_g, p, r, f])
            if rows:
                print(f"\\nSubgroup metrics by '{by}':")
                print("group, n, acc, prec, rec, f1")
                for r_ in rows:
                    print(",".join(str(x) for x in r_))

        for col in ["Gender", "Age"]:
            if col in val_meta.columns:
                # For Age, optionally bin
                if col == "Age":
                    try:
                        bins = [0, 20, 30, 40, 50, 100]
                        labels = ["<=20","21-30","31-40","41-50","50+"]
                        val_meta["AgeGroup"] = pd.cut(val_meta["Age"].astype(float), bins=bins, labels=labels, include_lowest=True)
                        subgroup_report(val_meta.rename(columns={"AgeGroup":"Age"}), "Age")
                    except Exception:
                        subgroup_report(val_meta, col)
                else:
                    subgroup_report(val_meta, col)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(),
                        "input_dim": int(X_train.shape[1])},
                       best_path)

    print("Best F1:", best_f1, "saved to", best_path)
    return best_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    fit_transform_save(df, artifacts_dir=args.artifacts_dir)
    train_model(artifacts_dir=args.artifacts_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)