"""Reproducible temporal ablation of Elliptic graph-neighbor aggregates.

Compares the 93 transaction-local columns (``feat_0`` through ``feat_92``)
against all 165 supplied Elliptic columns.  The latter additionally contains
the 72 neighbor-aggregate columns (``feat_93`` through ``feat_164``).

The model-selection and final-evaluation discipline deliberately mirrors
``train.py``: three rolling temporal validation folds choose tree count by
mean PR-AUC and choose a threshold from validation only; time steps 35..49
remain untouched until the final refit has completed.

Run from the repository root:

    $env:CHAINEYE_DATA_DIR = "C:\\path\\to\\elliptic_bitcoin_dataset"
    python app/ml/evaluate_feature_ablation.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parent.parent / "data" / "elliptic_bitcoin_dataset"
DATA_DIR = Path(os.environ.get("CHAINEYE_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
FEATURES_CSV = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_CSV = DATA_DIR / "elliptic_txs_classes.csv"
OUTPUT_PATH = HERE / "ablation_metrics.json"

N_FEATURES = 165
LOCAL_FEATURE_COLS = [f"feat_{index}" for index in range(93)]
ALL_FEATURE_COLS = [f"feat_{index}" for index in range(N_FEATURES)]
FEATURE_SETS = {
    "transaction_local_only": LOCAL_FEATURE_COLS,
    "all_features": ALL_FEATURE_COLS,
}
ROLLING_FOLDS = (
    (19, 20, 24),
    (24, 25, 29),
    (29, 30, 34),
)
ITERATION_CANDIDATES = (80, 120, 188, 260, 400, 600)
FINAL_TRAIN_END = 34
TEST_START = 35


def _scale_pos_weight(labels: np.ndarray) -> float:
    positive = int(labels.sum())
    negative = int((labels == 0).sum())
    return negative / max(positive, 1)


def _make_classifier(scale_pos_weight: float, n_estimators: int) -> lgb.LGBMClassifier:
    """Return the unchanged LightGBM configuration from train.py."""
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


def _select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.clip(thresholds[int(np.argmax(f1_scores))], 0.01, 0.99))


def _evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[1], average=None, zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "illicit_precision": float(precision[0]),
        "illicit_recall": float(recall[0]),
        "illicit_f1": float(f1[0]),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": {
            "tn_licit_licit": int(matrix[0, 0]),
            "fp_licit_illicit": int(matrix[0, 1]),
            "fn_illicit_licit": int(matrix[1, 0]),
            "tp_illicit_illicit": int(matrix[1, 1]),
        },
    }


def _load_labeled_data() -> pd.DataFrame:
    missing = [str(path) for path in (FEATURES_CSV, CLASSES_CSV) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Elliptic CSVs were not found. Set CHAINEYE_DATA_DIR to their directory: "
            + ", ".join(missing)
        )

    column_names = ["txId", "time_step"] + ALL_FEATURE_COLS
    dtypes: dict[str, str] = {"txId": "int64", "time_step": "int16"}
    dtypes.update({column: "float32" for column in ALL_FEATURE_COLS})
    print(f"[load] features: {FEATURES_CSV}")
    features = pd.read_csv(FEATURES_CSV, header=None, names=column_names, dtype=dtypes)
    classes = pd.read_csv(CLASSES_CSV, dtype={"txId": "int64", "class": "string"})
    data = features.merge(classes, on="txId", how="left").copy()
    data["label"] = data["class"].map({"1": 1, "2": 0})
    labeled = data[data["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype("int8")
    print(
        f"[load] labeled={len(labeled)} illicit={int(labeled['label'].sum())} "
        f"licit={int((labeled['label'] == 0).sum())}"
    )
    return labeled


def _run_feature_set(labeled: pd.DataFrame, name: str, columns: list[str]) -> dict:
    """Select only on rolling validation, then evaluate the future test once."""
    print(f"\n[ablation] {name}: {len(columns)} features")
    selection_results: dict[int, dict] = {}
    for n_estimators in ITERATION_CANDIDATES:
        folds = []
        for train_end, validation_start, validation_end in ROLLING_FOLDS:
            fold_train = labeled[labeled["time_step"] <= train_end]
            fold_validation = labeled[
                labeled["time_step"].between(validation_start, validation_end)
            ]
            train_labels = fold_train["label"].to_numpy()
            validation_labels = fold_validation["label"].to_numpy()
            model = _make_classifier(_scale_pos_weight(train_labels), n_estimators)
            model.fit(fold_train[columns].to_numpy(), train_labels)
            probabilities = model.predict_proba(fold_validation[columns].to_numpy())[:, 1]
            fold_threshold = _select_f1_threshold(validation_labels, probabilities)
            folds.append(
                {
                    "train": f"time_step 1..{train_end}",
                    "validation": f"time_step {validation_start}..{validation_end}",
                    "n_train": int(len(train_labels)),
                    "n_validation": int(len(validation_labels)),
                    "threshold": fold_threshold,
                    **_evaluate(validation_labels, probabilities, fold_threshold),
                }
            )
        mean_pr_auc = float(np.mean([fold["pr_auc"] for fold in folds]))
        selection_results[n_estimators] = {"mean_pr_auc": mean_pr_auc, "folds": folds}
        print(f"  trees={n_estimators:3d} mean_validation_pr_auc={mean_pr_auc:.6f}")

    best_iteration = max(
        ITERATION_CANDIDATES,
        key=lambda candidate: selection_results[candidate]["mean_pr_auc"],
    )
    selected_folds = selection_results[best_iteration]["folds"]
    threshold = float(np.median([fold["threshold"] for fold in selected_folds]))

    final_train = labeled[labeled["time_step"] <= FINAL_TRAIN_END]
    test = labeled[labeled["time_step"] >= TEST_START]
    train_labels = final_train["label"].to_numpy()
    test_labels = test["label"].to_numpy()
    scale_pos_weight = _scale_pos_weight(train_labels)
    final_model = _make_classifier(scale_pos_weight, best_iteration)
    final_model.fit(final_train[columns].to_numpy(), train_labels)
    test_probabilities = final_model.predict_proba(test[columns].to_numpy())[:, 1]
    test_metrics = _evaluate(test_labels, test_probabilities, threshold)
    print(
        f"[test] trees={best_iteration} threshold={threshold:.6f} "
        f"F1={test_metrics['illicit_f1']:.6f} PR-AUC={test_metrics['pr_auc']:.6f}"
    )

    return {
        "feature_columns": columns,
        "feature_count": len(columns),
        "model_selection": {
            "metric": "mean rolling-validation PR-AUC",
            "iteration_candidates": list(ITERATION_CANDIDATES),
            "candidate_mean_pr_auc": {
                str(candidate): result["mean_pr_auc"]
                for candidate, result in selection_results.items()
            },
            "selected_folds": selected_folds,
            "best_iteration": best_iteration,
        },
        "threshold": threshold,
        "threshold_selection": "median of fold-specific illicit-F1 optima on rolling validation",
        "scale_pos_weight": float(scale_pos_weight),
        "n_train": int(len(train_labels)),
        "n_test": int(len(test_labels)),
        "untouched_test": test_metrics,
    }


def main() -> None:
    started = time.monotonic()
    labeled = _load_labeled_data()
    results = {
        name: _run_feature_set(labeled, name, columns)
        for name, columns in FEATURE_SETS.items()
    }
    local = results["transaction_local_only"]["untouched_test"]
    all_features = results["all_features"]["untouched_test"]
    metrics = {
        "evaluation": "Elliptic graph-neighbor aggregate feature ablation",
        "feature_semantics": {
            "transaction_local_only": "feat_0..feat_92 (93 transaction-local features)",
            "neighbor_aggregates_added": "feat_93..feat_164 (72 graph-neighbor aggregate features)",
            "all_features": "feat_0..feat_164 (165 features)",
        },
        "evaluation_protocol": {
            "selection": "tree count by mean PR-AUC across rolling temporal validation folds; threshold from validation only",
            "rolling_validation": [
                {"train": f"time_step 1..{train_end}", "validation": f"time_step {start}..{end}"}
                for train_end, start, end in ROLLING_FOLDS
            ],
            "final_train": "time_step 1..34",
            "untouched_test": "time_step 35..49; evaluated once per feature set after selection and final refit",
            "model_configuration": "LightGBM settings and candidates identical to app/ml/train.py",
            "random_state": 42,
        },
        "feature_sets": results,
        "untouched_test_difference_all_minus_local": {
            metric: float(all_features[metric] - local[metric])
            for metric in ("illicit_precision", "illicit_recall", "illicit_f1", "pr_auc", "roc_auc")
        },
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
        output.write("\n")
    print(f"\n[done] wrote {OUTPUT_PATH} in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
