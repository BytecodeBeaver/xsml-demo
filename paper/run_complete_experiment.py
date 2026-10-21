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
import pandas as pd
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

def run_experiment(artifacts_dir: Path):
    experiment_pipeline = HintExperiment(
        target_set="any",
        include_elapsed_seconds=True,
        quick=False,
        load_artifacts=False,
        test_size=TEST_SIZE,
        artifacts_dir=artifacts_dir,
    )
    return experiment_pipeline.run()

def _generate_compact_summary(summary_dir: Path, experiment_results):
    compact_summary = experiment_results.reset_index()
    metric_labels = {
        "accuracy": "Średnie accuracy",
        "precision": "Średnie precision",
        "recall": "Średnie recall",
        "f1": "Średnie F1",
    }
    for metric, label in metric_labels.items():
        compact_summary[label] = compact_summary.apply(
            lambda row: (
                f"{row[f'cv_{metric}']:.3f}".replace(".", ",")
                + " ±"
                + f"{row[f'cv_{metric}_std']:.3f}".replace(".", ",")
            ),
            axis=1,
        )

    compact_summary = compact_summary.rename(
        columns={"model": "Model", "representation": "Reprezentacja"}
    )[["Model", "Reprezentacja", *metric_labels.values()]]
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "models_means_std.csv"
    compact_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
        sep=";",
        decimal=",",
    )
    return summary_path

def _generate_validation_summary(summary_dir: Path, experiment_results):
    validation_summary = experiment_results.reset_index()
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    for metric, label in metric_labels.items():
        validation_summary[label] = validation_summary[f"test_{metric}"].map(
            lambda value: f"{value:.3f}".replace(".", ",")
        )

    validation_summary = validation_summary.rename(
        columns={"model": "Model", "representation": "Reprezentacja"}
    )[["Model", "Reprezentacja", *metric_labels.values()]]
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "models_validation.csv"
    validation_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
        sep=";",
    )
    return summary_path

def _generate_fold_summaries(summary_dir: Path, experiment_results):
    fold_rows = []
    for _, result in experiment_results.iterrows():
        summary_path = Path(result["summary_path"])
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        with summary_path.open(encoding="utf-8") as summary_file:
            model_summary = json.load(summary_file)

        for fold in model_summary["cv_fold_results"]:
            fold_rows.append(
                {
                    "Model": model_summary["model"],
                    "Reprezentacja": model_summary["representation"],
                    "Fold": fold["fold"],
                    **fold["metrics"],
                }
            )

    fold_summary = pd.DataFrame(fold_rows)
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    for metric, label in metric_labels.items():
        fold_summary[label] = fold_summary[metric].map(
            lambda value: f"{value:.3f}".replace(".", ",")
        )

    summary_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for (model_name, representation), model_rows in fold_summary.groupby(
        ["Model", "Reprezentacja"], sort=False
    ):
        model_rows = model_rows.sort_values(["Reprezentacja", "Fold"])
        model_rows = model_rows[
            ["Model", "Reprezentacja", "Fold", *metric_labels.values()]
        ]
        summary_path = summary_dir / f"{model_name}_{representation}_folds.csv"
        model_rows.to_csv(
            summary_path,
            index=False,
            encoding="utf-8",
            sep=";",
        )
        output_paths.append(summary_path)
    return output_paths

def _generate_validation_confusion_matrices(summary_dir: Path, experiment_results):
    matrix_dir = summary_dir.parent / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for _, result in experiment_results.iterrows():
        summary_path = Path(result["summary_path"])
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        with summary_path.open(encoding="utf-8") as summary_file:
            model_summary = json.load(summary_file)

        confusion = np.asarray(
            model_summary["holdout_confusion_matrix"]["values"]
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=["No hint (0)", "Hint (1)"],
        )
        display.plot(cmap="Blues", values_format="d", colorbar=False)
        display.ax_.set_title(
            f"Validation set — {model_summary['model']} / "
            f"{model_summary['representation']}"
        )
        display.figure_.tight_layout()

        image_path = matrix_dir / (
            f"{model_summary['model']}_{model_summary['representation']}.png"
        )
        display.figure_.savefig(image_path, dpi=160, bbox_inches="tight")
        plt.close(display.figure_)
        output_paths.append(image_path)

    return output_paths

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
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    metric_colors = {
        "accuracy": "#4C78A8",
        "precision": "#F58518",
        "recall": "#54A24B",
        "f1": "#E45756",
    }

    results = experiment_results.reset_index()
    available_models = [
        model for model in model_order if model in results["model"].unique()
    ]
    if not available_models:
        return

    figure, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True, squeeze=False)
    bar_width = 0.18
    x_positions = np.arange(len(available_models))

    for axis, representation in zip(axes[0], representation_order):
        representation_results = results[
            results["representation"] == representation
        ].set_index("model")
        for metric_index, (metric_name, metric_label) in enumerate(metric_labels.items()):
            metric_column = f"test_{metric_name}"
            values = [
                representation_results.at[model, metric_column]
                if model in representation_results.index
                else np.nan
                for model in available_models
            ]
            offsets = (metric_index - (len(metric_labels) - 1) / 2) * bar_width
            bars = axis.bar(
                x_positions + offsets,
                values,
                width=bar_width,
                label=metric_label,
                color=metric_colors[metric_name],
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.4f}" if np.isfinite(value) else "" for value in values],
                padding=3,
                fontsize=7,
                rotation=90,
            )

        axis.set_title(f"Knowledge representation: {representation}")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [model_labels[model] for model in available_models],
            rotation=20,
            ha="right",
        )
        axis.set_ylim(0, 1.15)
        axis.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 0].set_ylabel("Holdout score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Metric",
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
    )
    figure.suptitle("Holdout metrics by model and knowledge representation")
    figure.tight_layout(rect=(0, 0, 0.86, 0.95))
    figure.savefig(metrics_dir / "all_metrics.png", dpi=160, bbox_inches="tight")
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
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_feature_extraction()
    experiment_dir = Path("data/experiments", run_id)
    experiment_results = run_experiment(artifacts_dir=experiment_dir)
    compact_summary_path = _generate_compact_summary(
        summary_dir=experiment_dir / "compact_summary",
        experiment_results=experiment_results,
    )
    validation_summary_path = _generate_validation_summary(
        summary_dir=experiment_dir / "compact_summary",
        experiment_results=experiment_results,
    )
    fold_summary_paths = _generate_fold_summaries(
        summary_dir=experiment_dir / "compact_summary",
        experiment_results=experiment_results,
    )
    _generate_validation_confusion_matrices(
        summary_dir=experiment_dir / "compact_summary",
        experiment_results=experiment_results,
    )
    generate_charts(
        charts_dir=Path("data/charts", run_id),
        experiment_results=experiment_results,
    )
    print(f"Compact summary saved to: {compact_summary_path}")
    print(f"Validation metrics saved to: {validation_summary_path}")
    print(f"Fold metrics saved to: {', '.join(map(str, fold_summary_paths))}")
    print(
        "Validation confusion matrices saved to: "
        f"{experiment_dir / 'confusion_matrices'}"
    )

if __name__ == "__main__":
    main()