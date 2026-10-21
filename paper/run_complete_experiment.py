"""
Ten skrypt uruchamia kompletny eksperyment w jednolity sposób.

Uruchamia zarówno ekstraktor cech w oparciu o obecną wersję pliku XSML z misją hardening_linux
oraz eksperyment, który odpowiednio dzieli zbiór i trenuje model ML na tych danych.
"""

from pathlib import Path
from datetime import datetime
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.training.experiment import HintExperiment
from parser.feature_extractor import FeatureExtractionPipeline

TEST_SIZE = 0.2

def run_feature_extraction():
    feature_extraction_pipeline = FeatureExtractionPipeline(
        mission=Path("data/missions/hardening_linux.yaml"),
        output_dir=Path("data/features"),
        prefix=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        random_seed=42,
        inventory_noise_variants=2,
        max_states=10_000,
    )
    return feature_extraction_pipeline.run()

def run_experiment():
    experiment_pipeline = HintExperiment(
        target_set="any",
        include_elapsed_seconds=True,
        quick=True,
        load_artifacts=False,
        test_size=TEST_SIZE,
        artifacts_dir=Path("data/experiments", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
    )
    return experiment_pipeline.run()

def _generate_confusion_matrixes(charts_dir: Path, experiment_results):
    confusion_matrix_dir = charts_dir / "confusion_matrix"
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)

    model_order = [
        "baseline_xgb",
        "baseline_decision_tree",
        "xgboost",
        "decision_tree",
    ]
    representation_order = ["flat", "xsml"]
    matrices = {}

    for _, result in experiment_results.iterrows():
        summary_path = Path(result["summary_path"])
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        with summary_path.open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)

        model_name = summary["model"]
        representation = summary["representation"]
        matrix = np.asarray(summary["holdout_confusion_matrix"]["values"])
        matrices[(model_name, representation)] = matrix

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["No hint (0)", "Hint (1)"],
        )
        display.plot(cmap="Blues", values_format="d", colorbar=False)
        display.ax_.set_title(f"{model_name} — {representation}")
        display.figure_.tight_layout()

        representation_dir = confusion_matrix_dir / representation
        representation_dir.mkdir(parents=True, exist_ok=True)
        display.figure_.savefig(
            representation_dir / f"{model_name}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(display.figure_)

    model_groups = {
        "baseline": [model for model in model_order if model.startswith("baseline_")],
        "tuned": [model for model in model_order if not model.startswith("baseline_")],
    }
    for group_name, group_models in model_groups.items():
        available_models = [
            model for model in group_models
            if any((model, representation) in matrices for representation in representation_order)
        ]
        if not available_models:
            continue

        figure, axes = plt.subplots(
            len(available_models),
            len(representation_order),
            figsize=(12, 4 * len(available_models)),
            squeeze=False,
        )
        for row_index, model_name in enumerate(available_models):
            for column_index, representation in enumerate(representation_order):
                axis = axes[row_index, column_index]
                matrix = matrices.get((model_name, representation))
                if matrix is None:
                    axis.set_visible(False)
                    continue
                display = ConfusionMatrixDisplay(
                    confusion_matrix=matrix,
                    display_labels=["No hint (0)", "Hint (1)"],
                )
                display.plot(
                    ax=axis,
                    cmap="Blues",
                    values_format="d",
                    colorbar=False,
                )
                axis.set_title(f"{model_name} — {representation}")

        figure.suptitle(f"Holdout confusion matrices — {group_name} models")
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        figure.savefig(
            confusion_matrix_dir / f"{group_name}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

def _generate_metric_charts(charts_dir: Path, experiment_results):
    metrics_dir = charts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_order = [
        "baseline_xgb",
        "baseline_decision_tree",
        "xgboost",
        "decision_tree",
    ]
    model_labels = {
        "baseline_xgb": "Baseline XGBoost",
        "baseline_decision_tree": "Baseline Decision Tree",
        "xgboost": "Tuned XGBoost",
        "decision_tree": "Tuned Decision Tree",
    }
    representation_order = ["flat", "xsml"]
    representation_colors = {"flat": "#4C78A8", "xsml": "#F58518"}
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }

    results = experiment_results.reset_index()
    x_positions = np.arange(len(model_order))
    bar_width = 0.36

    for metric_name, metric_label in metric_labels.items():
        metric_column = f"test_{metric_name}"
        figure, axis = plt.subplots(figsize=(10, 6))

        for representation_index, representation in enumerate(representation_order):
            representation_results = results[
                results["representation"] == representation
            ].set_index("model")
            values = [
                representation_results.at[model, metric_column]
                if model in representation_results.index
                else np.nan
                for model in model_order
            ]
            offsets = (
                representation_index - (len(representation_order) - 1) / 2
            ) * bar_width
            bars = axis.bar(
                x_positions + offsets,
                values,
                width=bar_width,
                label=representation,
                color=representation_colors[representation],
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.4f}" if np.isfinite(value) else "" for value in values],
                padding=3,
                fontsize=8,
            )

        axis.set_title(f"{metric_label} by training method and representation")
        axis.set_ylabel(metric_label)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([model_labels[model] for model in model_order], rotation=15, ha="right")
        axis.set_ylim(0, 1.12)
        axis.legend(
            title="Knowledge representation",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        figure.tight_layout(rect=(0, 0, 0.82, 1))
        figure.savefig(metrics_dir / f"{metric_name}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)

def generate_charts(charts_dir: Path, experiment_results):
    _generate_confusion_matrixes(
        charts_dir=charts_dir,
        experiment_results=experiment_results,
    )
    _generate_metric_charts(
        charts_dir=charts_dir,
        experiment_results=experiment_results,
    )

def main():
    run_feature_extraction()
    experiment_results = run_experiment()
    generate_charts(
        charts_dir=Path("data/charts", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
        experiment_results=experiment_results,
    )

if __name__ == "__main__":
    main()