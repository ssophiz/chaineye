# ChainEye — GCN Graph Model (Elliptic Bitcoin AML)

A standalone **Graph Convolutional Network (GCN)** node classifier trained on the
Elliptic Bitcoin transaction graph. It is a **differentiator** for the 2026 금융 AI
Challenge writeup: unlike the LightGBM baseline, which treats each transaction as an
independent feature vector, the GCN **propagates features across the transaction graph**,
so its prediction for a node also depends on that node's neighbours — capturing the
"who transacts with whom" structural signal that gradient-boosted trees cannot see.

This model is **not wired into inference.py or the backend** — it is a research artifact
that produces a trained model, metrics, and this narrative.

## Setup

```
graph      : 203,769 nodes (all transactions), 468,710 undirected edges
features   : the 165 Elliptic node features (standardized on train-mask stats)
labels     : illicit=1, licit=0, unknown=masked out of the loss
split       : TEMPORAL — train = labeled nodes with time_step 1..34
                          test  = labeled nodes with time_step 35..49
                          (identical protocol to the LightGBM baseline)
architecture: 3 × GCNConv (165 → 128 → 128 → 2), ReLU, dropout 0.3
loss        : class-weighted cross-entropy (illicit weight ≈ 7.63, inverse-frequency)
training    : 400 epochs, Adam (lr 0.01, wd 5e-4), full-batch on CPU (~10 min)
```

Reproduce:

```
C:/Users/DELL/fsec-ai-challenge-2026/.venv/Scripts/python.exe app/ml/gcn/gcn_train.py
```

## Test-set metrics (temporal split, illicit = positive class)

| Model                 | Illicit Precision | Illicit Recall | Illicit F1 | ROC-AUC |
|-----------------------|:-----------------:|:--------------:|:----------:|:-------:|
| LightGBM (baseline)   |      0.894        |     0.732      | **0.805**  | **0.932** |
| **GCN (this model)**  |      0.561        |     0.563      | **0.562**  | **0.892** |

Test set: 16,670 labeled nodes (1,083 illicit / 15,587 licit).
Confusion matrix (rows = true [licit, illicit], cols = predicted):

```
[[15109   478]
 [  473   610]]
```

Full metrics in [`gcn_metrics.json`](gcn_metrics.json).

## Comparison to LightGBM

On the same temporal split, the GCN reaches **illicit-F1 0.562 and ROC-AUC 0.892**,
below the LightGBM baseline's **0.805 / 0.932**. This gap is the well-documented pattern
on Elliptic: because the 165 hand-engineered features are already highly discriminative,
gradient-boosted trees exploit them directly, whereas a plain GCN smooths each node's
signal against its neighbourhood — which helps recall on structurally-embedded illicit
clusters but dilutes the sharp per-node cues and hurts precision, especially on the later
time steps where illicit actors adopt new patterns (temporal distribution shift hits the
graph model harder). The value of the GCN in the ChainEye narrative is **complementary,
not competitive**: it encodes *relational* risk — a transaction looking suspicious because
of the company it keeps — which is exactly the money-laundering typology (layering, peel
chains, mixer fan-out) that a per-transaction tree model is blind to. The strong ROC-AUC
(0.892) confirms the graph carries real signal; the natural next step for the competition
would be to **ensemble** the GCN's structural score with the LightGBM baseline, or upgrade
to a GraphSAGE/GAT with skip-connections, rather than replacing the tree model.

## Files

| File              | Description                                               |
|-------------------|-----------------------------------------------------------|
| `gcn_train.py`    | End-to-end training script (data → graph → train → eval). |
| `gcn_model.pt`    | Trained weights (state_dict) + config + feature scaler.   |
| `gcn_metrics.json`| Test-set metrics and run metadata.                        |
| `README.md`       | This file.                                                |

### Loading the saved model

```python
import torch
from gcn_train import GCN
ckpt = torch.load("app/ml/gcn/gcn_model.pt", weights_only=False)
cfg = ckpt["config"]
model = GCN(cfg["in_dim"], cfg["hidden"], cfg["n_classes"], cfg["dropout"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
# standardize new features with ckpt["feat_mean"] / ckpt["feat_std"] before forward()
```

*Note:* this is a fixed 400-epoch exploratory checkpoint. Training logs no longer inspect
the future test mask each epoch. Any future architecture or early-stopping selection must
use a separate temporal validation window before the final test evaluation.
