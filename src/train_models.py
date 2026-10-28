"""Train transparent baselines with year-based, leakage-aware evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "goes18_forecast_features.csv"
OUT = ROOT / "reports"
FIGURES = OUT / "figures"
TARGET = "target_m_or_x_start_next_60m"
FEATURES = [
    "log10_xrsb_now",
    "log10_xrsa_now",
    "log10_hardness_ratio",
    "log10_xrsb_median_10m",
    "log10_xrsb_max_30m",
    "log10_xrsb_std_30m",
    "log10_xrsb_change_10m",
]


def choose_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "precision_at_threshold": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, prediction, zero_division=0)),
        "alert_rate": float(prediction.mean()),
        "positive_rate": float(y_true.mean()),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["time_utc"]).set_index("time_utc")
    # Each label looks 60 minutes ahead. Drop rows whose target horizon crosses split boundaries.
    train_end = pd.Timestamp("2024-01-01", tz="UTC")
    valid_end = pd.Timestamp("2025-01-01", tz="UTC")
    horizon = pd.Timedelta(minutes=60)
    train = frame.loc[frame.index < train_end - horizon]
    valid = frame.loc[(frame.index >= train_end) & (frame.index < valid_end - horizon)]
    test = frame.loc[(frame.index >= valid_end) & (frame.index < pd.Timestamp("2026-01-01", tz="UTC") - horizon)]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Expected 2022-2023 train, 2024 validation, and 2025 test rows.")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_valid, y_valid = valid[FEATURES], valid[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    models = {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", C=0.5),
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.07, max_leaf_nodes=15,
            min_samples_leaf=60, l2_regularization=1.0, random_state=42,
        ),
    }

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    curves = {}
    predictions = pd.DataFrame(index=test.index)
    predictions["observed_m_or_x_start_next_60m"] = y_test
    for name, model in models.items():
        print(f"Fitting {name}...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        valid_probability = model.predict_proba(X_valid)[:, 1]
        threshold = choose_f1_threshold(y_valid.to_numpy(), valid_probability)
        test_probability = model.predict_proba(X_test)[:, 1]
        result = {
            "validation": metrics(y_valid.to_numpy(), valid_probability, threshold),
            "test_2025": metrics(y_test.to_numpy(), test_probability, threshold),
        }
        all_results[name] = result
        predictions[name.lower().replace(" ", "_")] = test_probability
        curves[name] = precision_recall_curve(y_test, test_probability)
        print(f"  test average precision={result['test_2025']['average_precision']:.4f}; "
              f"recall={result['test_2025']['recall_at_threshold']:.3f}; "
              f"precision={result['test_2025']['precision_at_threshold']:.3f}", flush=True)

    base_rate = float(y_test.mean())
    payload = {
        "target": "M-class-or-stronger GOES flare start within the next 60 minutes",
        "features": FEATURES,
        "split": {"train": "2022-06 to 2023", "validation": "2024", "test": "2025"},
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "test_positive_rate_baseline_average_precision": base_rate,
        "models": all_results,
        "interpretation": "Exploratory retrospective result. Overlapping time windows create dependence; performance is not an operational warning-system estimate.",
    }
    (OUT / "model_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    predictions.to_csv(ROOT / "data" / "processed" / "test_predictions_2025.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (precision, recall, _) in curves.items():
        ap = all_results[name]["test_2025"]["average_precision"]
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.axhline(base_rate, color="gray", linestyle="--", label=f"Prevalence baseline ({base_rate:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="M/X flare onset forecast: untouched 2025 test set")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25); ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "precision_recall_2025.png", dpi=180)
    plt.close(fig)

    lines = [
        "# First model evaluation",
        "",
        "## Evaluation design",
        "",
        "- Train: June 2022–2023. Validation: 2024. Untouched test: 2025.",
        "- Rows within 60 minutes of a year boundary are excluded so target horizons do not cross splits.",
        "- The target is an NOAA catalog M/X event start in the next hour. Features contain only past/current X-ray measurements.",
        "- The decision threshold is chosen on validation data to maximize F1 and then frozen for 2025.",
        "- Average precision is compared with the 2025 positive-class prevalence baseline.",
        "",
        "## Results on 2025 test period",
        "",
        "| Model | AP (baseline shown below) | ROC-AUC | Precision | Recall | F1 | Alert rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in all_results.items():
        m = result["test_2025"]
        lines.append(f"| {name} | {m['average_precision']:.3f} | {m['roc_auc']:.3f} | {m['precision_at_threshold']:.3f} | {m['recall_at_threshold']:.3f} | {m['f1_at_threshold']:.3f} | {m['alert_rate']:.2%} |")
    lines += [
        "",
        f"2025 event-window prevalence baseline AP: **{base_rate:.3f}**.",
        "",
        "![Precision-recall curves](figures/precision_recall_2025.png)",
        "",
        "## Limitations",
        "",
        "This is retrospective classification of overlapping 5-minute samples, not event-independent operational forecasting. The one-hour target horizon, threshold, alert burden, and event-level scoring need sensitivity analysis before making scientific or operational claims. The test period is a single solar-cycle year.",
    ]
    (OUT / "first_model_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {OUT / 'first_model_evaluation.md'}", flush=True)


if __name__ == "__main__":
    main()
