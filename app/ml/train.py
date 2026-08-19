"""
ChainEye (체인아이) - Bitcoin money-laundering detection
Training script for the 2026 금융 AI Challenge.

Trains a LightGBM binary classifier on the Elliptic dataset with a strict
temporal protocol:

    rolling validation = (train 1..19, validate 20..24),
                         (train 1..24, validate 25..29),
                         (train 1..29, validate 30..34)
    final train        = time_step 1..34
    untouched test     = time_step 35..49

Tree count and the classification threshold are selected on rolling validation
only. The final model is then refit on time steps 1..34 and evaluated once on
the untouched test window.

Run:
    C:/Users/DELL/fsec-ai-challenge-2026/.venv/Scripts/python.exe app/ml/train.py

Artifacts written into app/ml/:
    - chaineye_model.pkl        : trained LightGBM classifier (joblib)
    - feature_table.parquet     : features indexed by txId (fast inference lookup)
    - edges.parquet             : directed edge list (txId1 -> txId2)
    - shap_background.parquet    : small background sample for SHAP at inference
    - metrics.json              : evaluation metrics on the test set
    - feature_importance.csv     : global SHAP feature importance
"""
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(HERE, "..", "..", "data", "elliptic_bitcoin_dataset")
)
DATA_DIR = os.path.abspath(os.environ.get("CHAINEYE_DATA_DIR", DEFAULT_DATA_DIR))
FEATURES_CSV = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
CLASSES_CSV = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")
EDGES_CSV = os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv")

N_FEATURES = 165
FEATURE_COLS = [f"feat_{i}" for i in range(N_FEATURES)]  # feat_0 .. feat_164
ROLLING_FOLDS = (
    (19, 20, 24),
    (24, 25, 29),
    (29, 30, 34),
)
ITERATION_CANDIDATES = (80, 120, 188, 260, 400, 600)
FINAL_TRAIN_END = 34
TEST_START = 35


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


def _make_classifier(scale_pos_weight: float, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Select the illicit-class F1 threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    return float(np.clip(thresholds[best_index], 0.01, 0.99))


def _evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "illicit_precision": float(p[0]),
        "illicit_recall": float(r[0]),
        "illicit_f1": float(f1[0]),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {
            "tn_licit_licit": int(cm[0, 0]),
            "fp_licit_illicit": int(cm[0, 1]),
            "fn_illicit_licit": int(cm[1, 0]),
            "tp_illicit_illicit": int(cm[1, 1]),
        },
    }


def load_data():
    print(f"[load] reading features from {FEATURES_CSV} ...")
    t0 = time.time()
    # features.csv has NO header. col0=txId, col1=time_step, col2..166 = 165 feats.
    # Read txId + time_step as needed dtypes; feats as float32 to save memory.
    col_names = ["txId", "time_step"] + FEATURE_COLS
    dtype = {"txId": "int64", "time_step": "int16"}
    for c in FEATURE_COLS:
        dtype[c] = "float32"
    feats = pd.read_csv(FEATURES_CSV, header=None, names=col_names, dtype=dtype)
    print(f"[load] features shape={feats.shape} in {time.time()-t0:.1f}s")

    classes = pd.read_csv(CLASSES_CSV, dtype={"txId": "int64", "class": "string"})
    print(f"[load] classes shape={classes.shape}")

    # Consolidate blocks after the wide merge so adding the label column does
    # not trigger pandas' highly-fragmented DataFrame warning.
    df = feats.merge(classes, on="txId", how="left").copy()
    return df, feats


def main():
    df, feats = load_data()

    # Map labels: illicit "1" -> 1, licit "2" -> 0, else unknown (drop for supervised)
    label_map = {"1": 1, "2": 0}
    df["label"] = df["class"].map(label_map)
    labeled = df[df["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype("int8")
    print(f"[prep] labeled rows={len(labeled)} "
          f"illicit={int((labeled.label==1).sum())} licit={int((labeled.label==0).sum())}")

    # Strict temporal split. Test is never passed to fitting or model
    # selection. Several historical rolling folds make selection less brittle
    # than relying on a single validation window.
    final_train_df = labeled[labeled["time_step"] <= FINAL_TRAIN_END]
    test_df = labeled[labeled["time_step"] >= TEST_START]

    X_train = final_train_df[FEATURE_COLS].values
    y_train = final_train_df["label"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values
    print(
        f"[split] final_train={X_train.shape} (illicit={int(y_train.sum())})  "
        f"test={X_test.shape} (illicit={int(y_test.sum())})"
    )

    # Stage 1: rolling temporal validation chooses the number of boosting trees
    # by mean PR-AUC. The serving threshold is the median of each fold's
    # illicit-F1 optimum, which is more robust than one window's optimum.
    print("[selection] rolling temporal validation ...")
    t0 = time.time()
    candidate_results = {}
    for n_estimators in ITERATION_CANDIDATES:
        fold_results = []
        for train_end, validation_start, validation_end in ROLLING_FOLDS:
            fold_train = labeled[labeled["time_step"] <= train_end]
            fold_validation = labeled[
                labeled["time_step"].between(validation_start, validation_end)
            ]
            X_fold_train = fold_train[FEATURE_COLS].values
            y_fold_train = fold_train["label"].values
            X_fold_validation = fold_validation[FEATURE_COLS].values
            y_fold_validation = fold_validation["label"].values

            fold_model = _make_classifier(
                _scale_pos_weight(y_fold_train), n_estimators
            )
            fold_model.fit(X_fold_train, y_fold_train)
            fold_proba = fold_model.predict_proba(X_fold_validation)[:, 1]
            fold_threshold = select_f1_threshold(y_fold_validation, fold_proba)
            fold_metrics = _evaluate(
                y_fold_validation, fold_proba, fold_threshold
            )
            fold_results.append({
                "train": f"time_step 1..{train_end}",
                "validation": (
                    f"time_step {validation_start}..{validation_end}"
                ),
                "n_train": int(len(y_fold_train)),
                "n_validation": int(len(y_fold_validation)),
                "threshold": fold_threshold,
                **fold_metrics,
            })

        mean_pr_auc = float(np.mean([fold["pr_auc"] for fold in fold_results]))
        candidate_results[n_estimators] = {
            "mean_pr_auc": mean_pr_auc,
            "folds": fold_results,
        }
        print(f"  trees={n_estimators:3d} mean_validation_pr_auc={mean_pr_auc:.4f}")

    best_iteration = max(
        ITERATION_CANDIDATES,
        key=lambda candidate: candidate_results[candidate]["mean_pr_auc"],
    )
    selected_folds = candidate_results[best_iteration]["folds"]
    threshold = float(np.median([fold["threshold"] for fold in selected_folds]))
    print(
        f"[selection] done in {time.time()-t0:.1f}s, "
        f"trees={best_iteration}, median_threshold={threshold:.4f}, "
        f"mean_pr_auc={candidate_results[best_iteration]['mean_pr_auc']:.4f}"
    )

    # Stage 2: refit on all pre-test data with the selected tree count. The
    # untouched test window is evaluated exactly once after fitting.
    scale_pos_weight = _scale_pos_weight(y_train)
    clf = _make_classifier(scale_pos_weight, best_iteration)
    print(
        "[final] refitting on time steps 1..34; "
        f"scale_pos_weight={scale_pos_weight:.2f} ..."
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[final] done in {time.time()-t0:.1f}s")

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test_metrics = _evaluate(y_test, proba, threshold)
    cm_values = test_metrics["confusion_matrix"]
    cm = np.array([
        [cm_values["tn_licit_licit"], cm_values["fp_licit_illicit"]],
        [cm_values["fn_illicit_licit"], cm_values["tp_illicit_illicit"]],
    ])

    print("\n==== UNTOUCHED TEST METRICS (illicit = positive) ====")
    print(f"Decision threshold: {threshold:.4f} (selected on validation)")
    print(f"Illicit Precision : {test_metrics['illicit_precision']:.4f}")
    print(f"Illicit Recall    : {test_metrics['illicit_recall']:.4f}")
    print(f"Illicit F1        : {test_metrics['illicit_f1']:.4f}   <-- headline")
    print(f"PR-AUC (AP)       : {test_metrics['pr_auc']:.4f}")
    print(f"ROC-AUC           : {test_metrics['roc_auc']:.4f}")
    print("Confusion matrix [rows=true licit/illicit, cols=pred licit/illicit]:")
    print(cm)
    print("\n" + classification_report(y_test, pred, target_names=["licit", "illicit"], digits=4))

    metrics = {
        **test_metrics,
        "evaluation_protocol": "rolling temporal validation and untouched future test",
        "split": {
            "rolling_validation": [
                {
                    "train": f"time_step 1..{train_end}",
                    "validation": f"time_step {validation_start}..{validation_end}",
                }
                for train_end, validation_start, validation_end in ROLLING_FOLDS
            ],
            "final_train": "time_step 1..34",
            "test": "time_step 35..49",
        },
        "model_selection": {
            "metric": "mean rolling-validation PR-AUC",
            "iteration_candidates": list(ITERATION_CANDIDATES),
            "candidate_mean_pr_auc": {
                str(candidate): result["mean_pr_auc"]
                for candidate, result in candidate_results.items()
            },
            "selected_folds": selected_folds,
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "best_iteration": best_iteration,
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": threshold,
        "threshold_selection": (
            "median of fold-specific illicit-F1 optima on rolling validation"
        ),
    }

    # ---- SHAP global importance ----
    print("[shap] computing global feature importance ...")
    import shap
    explainer = shap.TreeExplainer(clf)
    # sample test set for global importance (speed)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
    sv = explainer.shap_values(X_test[idx])
    if isinstance(sv, list):  # older shap returns list per class
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    fi = (pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs})
          .sort_values("mean_abs_shap", ascending=False))
    fi.to_csv(os.path.join(HERE, "feature_importance.csv"), index=False)
    print("[shap] top 10 features:")
    print(fi.head(10).to_string(index=False))

    # ---- Persist artifacts ----
    joblib.dump(clf, os.path.join(HERE, "chaineye_model.pkl"))

    # feature table indexed by txId (ALL nodes, so inference can score any node)
    feat_table = feats.set_index("txId")
    feat_table.to_parquet(os.path.join(HERE, "feature_table.parquet"))

    edges = pd.read_csv(EDGES_CSV, dtype={"txId1": "int64", "txId2": "int64"})
    edges.to_parquet(os.path.join(HERE, "edges.parquet"), index=False)

    # SHAP background: small representative sample of training features
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    bg = pd.DataFrame(X_train[bg_idx], columns=FEATURE_COLS)
    bg.to_parquet(os.path.join(HERE, "shap_background.parquet"), index=False)

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[done] artifacts written to", HERE)
    for fn in ["chaineye_model.pkl", "feature_table.parquet", "edges.parquet",
               "shap_background.parquet", "metrics.json", "feature_importance.csv"]:
        p = os.path.join(HERE, fn)
        print(f"   {fn:26s} {os.path.getsize(p)/1e6:8.2f} MB")


if __name__ == "__main__":
    main()
