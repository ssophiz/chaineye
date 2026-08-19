"""
ChainEye — GCN differentiation model (Bitcoin AML, 2026 금융 AI Challenge)
========================================================================

Standalone Graph Convolutional Network node classifier on the Elliptic
Bitcoin dataset. This is a *differentiator* alongside the LightGBM baseline:
it consumes the same 165 node features but also propagates them across the
transaction graph, so it can capture structural signal the GBT cannot see.

Protocol (identical to app/ml/train.py):
    - Nodes  : all 203,769 transactions, features = the 165 columns.
    - Edges  : elliptic_txs_edgelist.csv, used UNDIRECTED for message passing.
    - Labels : illicit "1" -> 1, licit "2" -> 0, "unknown" masked out of loss.
    - Split  : TEMPORAL — train = labeled nodes with time_step 1..34,
                          test  = labeled nodes with time_step 35..49.

Outputs (all under app/ml/gcn/):
    - gcn_model.pt     : trained model weights (state_dict) + config
    - gcn_metrics.json : test-set metrics (illicit P/R/F1, ROC-AUC, PR-AUC)

Run:
    C:/Users/DELL/fsec-ai-challenge-2026/.venv/Scripts/python.exe app/ml/gcn/gcn_train.py

Does NOT touch inference.py / train.py / any existing artifact.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path("C:/Users/DELL/fsec-ai-challenge-2026")
DATA = ROOT / "data" / "elliptic_bitcoin_dataset"
FEATURES_CSV = DATA / "elliptic_txs_features.csv"
CLASSES_CSV = DATA / "elliptic_txs_classes.csv"
EDGES_CSV = DATA / "elliptic_txs_edgelist.csv"

OUT_DIR = ROOT / "app" / "ml" / "gcn"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PT = OUT_DIR / "gcn_model.pt"
METRICS_JSON = OUT_DIR / "gcn_metrics.json"

FEATURE_COLS = [f"f{i}" for i in range(165)]

SEED = 42
EPOCHS = 400
HIDDEN = 128
DROPOUT = 0.3
LR = 0.01
WEIGHT_DECAY = 5e-4


# ----------------------------------------------------------------------------
# Data loading -> PyG Data object
# ----------------------------------------------------------------------------
def build_graph():
    t0 = time.time()
    # features.csv has NO header: col0=txId, col1=time_step, col2..166 = 165 feats
    col_names = ["txId", "time_step"] + FEATURE_COLS
    dtype = {"txId": "int64", "time_step": "int16"}
    for c in FEATURE_COLS:
        dtype[c] = "float32"
    feats = pd.read_csv(FEATURES_CSV, header=None, names=col_names, dtype=dtype)
    print(f"[load] features shape={feats.shape} in {time.time()-t0:.1f}s")

    classes = pd.read_csv(CLASSES_CSV, dtype={"txId": "int64", "class": "string"})
    label_map = {"1": 1, "2": 0}  # illicit -> 1, licit -> 0
    classes["label"] = classes["class"].map(label_map)  # NaN for "unknown"

    df = feats.merge(classes[["txId", "label"]], on="txId", how="left")

    # Contiguous node indexing in the row order of features.csv
    txid_to_idx = {tx: i for i, tx in enumerate(df["txId"].to_numpy())}
    n_nodes = len(df)

    # Node feature matrix
    x = torch.tensor(df[FEATURE_COLS].to_numpy(dtype=np.float32))

    # Labels: -1 where unknown (masked), else 0/1
    y_np = df["label"].to_numpy(dtype=np.float32)
    y = torch.tensor(np.where(np.isnan(y_np), -1, y_np).astype(np.int64))

    time_step = torch.tensor(df["time_step"].to_numpy(dtype=np.int64))

    # Edges -> undirected edge_index
    edges = pd.read_csv(EDGES_CSV, dtype={"txId1": "int64", "txId2": "int64"})
    src = edges["txId1"].map(txid_to_idx).to_numpy()
    dst = edges["txId2"].map(txid_to_idx).to_numpy()
    keep = ~(np.isnan(src) | np.isnan(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    edge_index = torch.tensor(np.vstack([src, dst]))
    edge_index = to_undirected(edge_index, num_nodes=n_nodes)
    print(f"[graph] nodes={n_nodes} undirected_edges={edge_index.size(1)}")

    # Masks: labeled AND in the temporal window
    labeled = y >= 0
    train_mask = labeled & (time_step <= 34)
    test_mask = labeled & (time_step >= 35)
    print(f"[split] train_labeled={int(train_mask.sum())} "
          f"(illicit={int(((y == 1) & train_mask).sum())})  "
          f"test_labeled={int(test_mask.sum())} "
          f"(illicit={int(((y == 1) & test_mask).sum())})")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


# ----------------------------------------------------------------------------
# Model: 3-layer GCN
# ----------------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x  # logits


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = build_graph()

    # Standardize features using TRAIN-mask statistics only (avoid leakage)
    x = data.x
    tr = data.train_mask
    mean = x[tr].mean(dim=0, keepdim=True)
    std = x[tr].std(dim=0, keepdim=True).clamp_min(1e-6)
    data.x = (x - mean) / std

    model = GCN(in_dim=data.x.size(1), hidden=HIDDEN, n_classes=2, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Class-weighted cross entropy (illicit is the rare positive)
    y_tr = data.y[tr]
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    # inverse-frequency weights, normalized so licit weight = 1
    w_pos = n_neg / max(n_pos, 1)
    class_weight = torch.tensor([1.0, w_pos], dtype=torch.float32)
    print(f"[weights] licit=1.0  illicit={w_pos:.2f}  (n_neg={n_neg}, n_pos={n_pos})")

    train_idx = tr
    test_idx = data.test_mask

    print(f"[train] {EPOCHS} epochs on CPU ...")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx], weight=class_weight)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            # Keep the future test mask untouched until the fixed 400-epoch
            # training run is complete. Model selection belongs on a separate
            # temporal validation mask, not on test telemetry.
            print(f"  epoch {epoch:4d}  train_loss={loss.item():.4f}")
    train_secs = time.time() - t0
    print(f"[train] done in {train_secs:.1f}s")

    # ---- Final evaluation on TEST mask ----
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        proba_all = F.softmax(logits, dim=1)[:, 1]
    yte = data.y[test_idx].cpu().numpy()
    proba = proba_all[test_idx].cpu().numpy()
    pred = (proba >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        yte, pred, labels=[1], average=None, zero_division=0)
    roc_auc = float(roc_auc_score(yte, proba))
    pr_auc = float(average_precision_score(yte, proba))
    cm = confusion_matrix(yte, pred, labels=[0, 1])

    print("\n==== GCN TEST-SET METRICS (temporal split, illicit = positive) ====")
    print(f"illicit  Precision={p[0]:.4f}  Recall={r[0]:.4f}  F1={f1[0]:.4f}")
    print(f"ROC-AUC={roc_auc:.4f}   PR-AUC={pr_auc:.4f}")
    print(f"confusion_matrix (rows true [licit, illicit], cols pred):\n{cm}")
    print("\n" + classification_report(
        yte, pred, target_names=["licit", "illicit"], digits=4, zero_division=0))

    metrics = {
        "model": "GCN (3x GCNConv, hidden=%d)" % HIDDEN,
        "dataset": "Elliptic Bitcoin",
        "split": "temporal (train time_step 1..34, test 35..49)",
        "n_nodes": int(data.x.size(0)),
        "n_undirected_edges": int(data.edge_index.size(1)),
        "n_train_labeled": int(train_idx.sum()),
        "n_test_labeled": int(test_idx.sum()),
        "epochs": EPOCHS,
        "class_weight_illicit": float(w_pos),
        "illicit_precision": float(p[0]),
        "illicit_recall": float(r[0]),
        "illicit_f1": float(f1[0]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "train_seconds": round(train_secs, 1),
        "lightgbm_baseline": {"illicit_f1": 0.8051, "roc_auc": 0.9317},
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\n[save] metrics -> {METRICS_JSON}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "in_dim": int(data.x.size(1)),
                "hidden": HIDDEN,
                "n_classes": 2,
                "dropout": DROPOUT,
                "arch": "3xGCNConv",
            },
            "feat_mean": mean,
            "feat_std": std,
            "metrics": metrics,
        },
        MODEL_PT,
    )
    print(f"[save] model -> {MODEL_PT}")
    return metrics


if __name__ == "__main__":
    main()
