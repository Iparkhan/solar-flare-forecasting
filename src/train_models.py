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


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float,
            probability_score: bool = True) -> dict[str, float | None]:
    prediction = probability >= threshold
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)) if probability_score else None,
        "threshold": float(threshold),
        "precision_at_threshold": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, prediction, zero_division=0)),
        "alert_rate": float(prediction.mean()),
        "positive_rate": float(y_true.mean()),
    }


def month_block_bootstrap(y: pd.Series, scores: dict[str, np.ndarray],
                          repetitions: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Resample whole calendar months to retain within-month time dependence."""
    month_labels = y.index.tz_localize(None).to_period("M")
    month_groups = [np.flatnonzero(month_labels == month) for month in month_labels.unique()]
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in scores}
    for _ in range(repetitions):
        selected = rng.integers(0, len(month_groups), size=len(month_groups))
        indices = np.concatenate([month_groups[i] for i in selected])
        target = y.to_numpy()[indices]
        if target.min() == target.max():
            continue
        for name, score in scores.items():
            samples[name].append(average_precision_score(target, score[indices]))
    baseline_name = "Single-feature XRS-B threshold baseline"
    rows = []
    for name, values in samples.items():
        values = np.asarray(values)
        row = {"model": name, "valid_replicates": len(values),
               "average_precision_median": float(np.median(values)),
               "ci_2_5": float(np.quantile(values, .025)),
               "ci_97_5": float(np.quantile(values, .975)),
               "delta_vs_xrsb_median": np.nan,
               "delta_ci_2_5": np.nan,
               "delta_ci_97_5": np.nan}
        if name != baseline_name and len(values):
            delta = values - np.asarray(samples[baseline_name])
            row.update({"delta_vs_xrsb_median": float(np.median(delta)),
                        "delta_ci_2_5": float(np.quantile(delta, .025)),
                        "delta_ci_97_5": float(np.quantile(delta, .975))})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA, parse_dates=["time_utc"]).set_index("time_utc")
    # Each label looks 60 minutes ahead. Drop rows whose target horizon crosses split boundaries.
    train_end = pd.Timestamp("2024-01-01", tz="UTC")
    valid_end = pd.Timestamp("2025-01-01", tz="UTC")
    horizon = pd.Timedelta("60min")
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

    # Rank only current XRS-B irradiance as a simple physical benchmark.
    train_flux = np.sort(X_train["log10_xrsb_now"].to_numpy())
    flux_scores = {
        "validation": np.searchsorted(train_flux, X_valid["log10_xrsb_now"].to_numpy(), side="right") / len(train_flux),
        "test": np.searchsorted(train_flux, X_test["log10_xrsb_now"].to_numpy(), side="right") / len(train_flux),
    }
    flux_threshold = choose_f1_threshold(y_valid.to_numpy(), flux_scores["validation"])
    baseline_name = "Single-feature XRS-B threshold baseline"
    all_results[baseline_name] = {
        "validation": metrics(y_valid.to_numpy(), flux_scores["validation"], flux_threshold, probability_score=False),
        "test_2025": metrics(y_test.to_numpy(), flux_scores["test"], flux_threshold, probability_score=False),
        "score_definition": "Empirical percentile rank of current log10 XRS-B in training data; rank is a score, not a probability.",
    }
    baseline_col = baseline_name.lower().replace(" ", "_")
    predictions[baseline_col] = flux_scores["test"]
    curves[baseline_name] = precision_recall_curve(y_test, flux_scores["test"])
    print(f"Fitting {baseline_name}...", flush=True)
    print(f"  test average precision={all_results[baseline_name]['test_2025']['average_precision']:.4f}; "
          f"recall={all_results[baseline_name]['test_2025']['recall_at_threshold']:.3f}; "
          f"precision={all_results[baseline_name]['test_2025']['precision_at_threshold']:.3f}", flush=True)

    bootstrap = month_block_bootstrap(y_test, {
        name: predictions[name.lower().replace(" ", "_")].to_numpy()
        for name in all_results
    })
    (OUT / "tables").mkdir(exist_ok=True)
    bootstrap.to_csv(OUT / "tables" / "monthly_block_bootstrap_ap_2025.csv", index=False, float_format="%.4f")

    base_rate = float(y_test.mean())
    payload = {
        "target": "M-class-or-stronger GOES flare start within the next 60 minutes",
        "features": FEATURES,
        "split": {"train": "2022-06 to 2023", "validation": "2024", "test": "2025"},
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "test_positive_rate_baseline_average_precision": base_rate,
        "models": all_results,
        "uncertainty": {
            "method": "Percentile interval from 1,000 bootstrap replicates sampling whole 2025 calendar months with replacement; preserves within-month dependence, but only 12 monthly blocks are available.",
            "results_csv": "tables/monthly_block_bootstrap_ap_2025.csv",
        },
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
        "## Monthly block-bootstrap uncertainty",
        "",
        "Intervals resample whole calendar months (1,000 replicates), retaining within-month dependence. Only 12 test blocks are available, so these intervals are approximate.",
        "",
        "| Model | AP median | 95% block-bootstrap interval | Paired AP delta vs XRS-B | 95% interval for delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in bootstrap.iterrows():
        delta = "—" if pd.isna(row["delta_vs_xrsb_median"]) else f"{row['delta_vs_xrsb_median']:+.3f}"
        delta_ci = "—" if pd.isna(row["delta_ci_2_5"]) else f"{row['delta_ci_2_5']:+.3f}–{row['delta_ci_97_5']:+.3f}"
        lines.append(f"| {row['model']} | {row['average_precision_median']:.3f} | {row['ci_2_5']:.3f}–{row['ci_97_5']:.3f} | {delta} | {delta_ci} |")
    lines += [
        "",
        "",
        "Paired differences use the same resampled months for each method and the XRS-B baseline. These intervals are more informative for model-versus-baseline comparison than separate AP intervals.",
        "",
        "Download: [`tables/monthly_block_bootstrap_ap_2025.csv`](tables/monthly_block_bootstrap_ap_2025.csv).",
        "",
        "## Limitations",
        "",
        "This is retrospective classification of overlapping five-minute samples, not an event-independent operational forecast. The single-feature XRS-B benchmark is an empirical percentile rank, not a probability. The complex model's AP advantage over the simple benchmark is small and the block-bootstrap intervals overlap; the simple benchmark has better event recall under this alarm rule. Event-level scores depend on the chosen threshold and episode rule, and uncertainty intervals use just 12 monthly groups. The test period is a single solar-cycle year.",
    ]
    (OUT / "first_model_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {OUT / 'first_model_evaluation.md'}", flush=True)


if __name__ == "__main__":
    main()
