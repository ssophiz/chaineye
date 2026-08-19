# ChainEye (체인아이) — ML module

Bitcoin money-laundering (illicit-transaction) detection for the 2026 금융 AI Challenge.
LightGBM binary classifier trained on the **Elliptic** dataset using rolling
**temporal validation** inside time steps 1..34 and an untouched future test
window at time steps 35..49.

## Live feature schema

The deployed LightGBM consumes 165 features for each transaction:

- **93 transaction-local features** describing the transaction itself.
- **72 graph-neighbor aggregate features** derived from its transaction-graph neighborhood.

This is a tabular model with graph context already included in its input vector; it is not the standalone end-to-end GCN in `app/ml/gcn/`.

## Test-set metrics (temporal split, illicit = positive class)

| Metric | Value |
|---|---|
| Illicit Precision | 0.8940 |
| Illicit Recall | 0.7322 |
| **Illicit F1 (headline)** | **0.8051** |
| PR-AUC (Average Precision) | 0.7995 |
| ROC-AUC | 0.9317 |

Confusion matrix (rows = true, cols = predicted), validation-selected threshold 0.5238:

|            | pred licit | pred illicit |
|------------|-----------:|-------------:|
| **true licit**   | 15493 | 94 |
| **true illicit** |   290 | 793 |

Train: 29,894 labeled tx (3,462 illicit). Test: 16,670 labeled tx (1,083 illicit).
`scale_pos_weight = 7.63` handles class imbalance. Tree count (600) is selected by
mean PR-AUC across three rolling validation folds; the decision threshold is the
median of the three fold-specific illicit-F1 optima. The future test window is not
used for fitting, tree-count selection, or threshold selection.

Top global SHAP features: feat_58, feat_52, feat_75, feat_57, feat_4, feat_124
(Elliptic features are anonymized; `feat_i` = column i of the 165 live features).

## Artifacts (all in `app/ml/`)

| File | Purpose |
|---|---|
| `chaineye_model.pkl` | trained LightGBM classifier (joblib) |
| `feature_table.parquet` | 165 live features (93 transaction-local + 72 graph-neighbor aggregates) indexed by txId for all 203,769 nodes |
| `edges.parquet` | directed edge list (txId1 -> txId2) |
| `shap_background.parquet` | background sample for SHAP |
| `feature_importance.csv` | global SHAP mean-abs importance |
| `metrics.json` | evaluation metrics |

## Inference API (`inference.py`) — imported by the backend

```python
from app.ml import inference
inference.load()                      # once at startup; idempotent
inference.score_tx("232629023")
#   {"txId": "232629023", "riskScore": 100, "label": "illicit",
#    "topFactors": [{"feature": "feat_1", "impact": 2.239}, ...]}  # up to 6, signed SHAP
inference.trace_tx("232629023", hops=2)
#   {"nodes": [{"id": str, "risk": int, "focus": bool}, ...],      # <= 60 nodes
#    "edges": [{"source": str, "target": str}, ...]}
```

- Unlabeled/unknown txId **with features** is still scored.
- txId **not in the feature table** returns `riskScore 0`, `label "licit"`, empty `topFactors` (never raises).
- `trace_tx` follows edges both directions up to `hops`; caps at ~60 nodes keeping highest-risk neighbors.

## GCN decision

The reproducible `evaluate_feature_ablation.py` run isolates the contribution of the 72 graph-neighbor aggregate features under the same rolling selection and untouched future test protocol.

| Feature set | Illicit F1 | PR-AUC |
|---|---:|---:|
| 93 transaction-local features | 0.7440 | 0.7858 |
| All 165 features | **0.8051** | **0.7995** |
| Absolute lift from graph aggregates | **+0.0611** | **+0.0137** |

`app/ml/gcn/` contains a separate standalone end-to-end GCN research run. It underperformed the graph-enhanced LightGBM on the temporal evaluation, so it is excluded from `inference.py` and the backend serving path. The feature ablation and GCN comparison answer different questions and are reported separately.

## Retrain

```powershell
$env:CHAINEYE_DATA_DIR = "C:\path\to\elliptic_bitcoin_dataset"
python app/ml/train.py
```

Re-reads the Elliptic CSVs from `CHAINEYE_DATA_DIR` (or the repository's
`data/elliptic_bitcoin_dataset/` default) and rebuilds all artifacts. The raw dataset is
excluded from Git and Docker; only the runtime artifacts are deployed.

Verify inference: `... python app/ml/verify_inference.py`
