"""Evaluate hint classifiers on the hardening mission.

Each hint is evaluated as an independent binary decision class, alongside an
"any hint shown" target formed by OR-ing all hint labels. Rows with the same
state but different idle-time snapshots collapse to one state row, positive
for a hint if any snapshot showed that hint. Both representations include
rule-derived action and goal predicates; XSML also has explicit per-goal
hierarchy features.

By default, ``elapsed_seconds`` is dropped because it directly encodes the
idle-time threshold used by hint rules. In that mode, duplicate state
snapshots are collapsed and positive if any snapshot showed the evaluated
hint. ``--include-elapsed-seconds`` keeps time as a predictor and preserves
each snapshot as a separate training row.

Stratified 5-fold CV on the training partition evaluates generalization to
unseen states from this mission; a separate stratified holdout is reserved for
final validation. Each outer training fold gets a separate 3-fold search.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Literal
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MISSION = "hardening_linux"
N_SPLITS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
ARTIFACT_VERSION = 2

# Small grids for the exhaustive, single-mission exploratory dataset.
XGBOOST_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.3],
}

DECISION_TREE_PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
}

SCORERS = {
    "accuracy": "accuracy",
    # zero_division=0: some folds/held-out missions have no positives to predict.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def _load_targets(mission: str) -> pd.DataFrame:
    """Load each hint target and add the OR label for any available hint."""
    targets = pd.read_csv(
        os.path.join(FEATURES_DIR, f"{mission}_targets.csv"),
        true_values=["true"],
        false_values=["false"],
    )
    hint_columns = [column for column in targets.columns if column.startswith("hint::")]
    labels = targets.set_index("sample_id")[hint_columns].astype(bool).astype(int)
    labels.columns = [column.removeprefix("hint::") for column in labels.columns]
    labels["any_hint"] = labels.any(axis=1).astype(int)
    return labels


def _load_xsml_features(mission: str) -> pd.DataFrame:
    frame = pd.read_csv(
        os.path.join(FEATURES_DIR, f"{mission}_xsml_features.csv"),
        true_values=["true"],
        false_values=["false"],
    )
    frame = frame.set_index("sample_id").drop(columns=["mission_id"])
    return frame.astype(float)


def _load_flat_features(mission: str) -> pd.DataFrame:
    """Keep Boolean state columns and one-hot encode non-Boolean categories."""
    frame = pd.read_csv(
        os.path.join(FEATURES_DIR, f"{mission}_flat_features.csv"),
        true_values=["true"],
        false_values=["false"],
    )
    frame = frame.set_index("sample_id").drop(columns=["mission_id"])
    state_columns = [column for column in frame.columns if column.startswith(("inventory.", "worldstate."))]
    categorical_columns = [column for column in state_columns if frame[column].dtype != bool]
    other_columns = [column for column in frame.columns if column not in categorical_columns]
    encoded = (
        pd.get_dummies(frame[categorical_columns], columns=categorical_columns)
        if categorical_columns
        else pd.DataFrame(index=frame.index)
    )
    return pd.concat([frame[other_columns], encoded], axis=1).astype(float)


def _load_mission(
    loader, mission: str, *, include_elapsed_seconds: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load hint labels; optionally collapse snapshots if time is excluded."""
    features, targets = loader(mission).align(_load_targets(mission), join="inner", axis=0)
    if include_elapsed_seconds:
        return features, targets

    features = features.drop(columns=["elapsed_seconds"])
    feature_columns = list(features.columns)
    merged = features.join(targets)
    collapsed = merged.groupby(
        feature_columns, as_index=False, sort=False
    )[list(targets.columns)].max()
    return collapsed[feature_columns], collapsed[list(targets.columns)]


def _safe_name(value: str) -> str:
    """Return a filename-safe identifier for an artifact component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _artifact_paths(
    artifact_dir: str, target_name: str, model_name: str, representation: str
) -> tuple[str, str]:
    key = "__".join(map(_safe_name, (target_name, model_name, representation)))
    return (
        os.path.join(artifact_dir, "models", f"{key}.joblib"),
        os.path.join(artifact_dir, "splits", "__".join(map(_safe_name, (target_name, representation)))),
    )


def _make_splits(
    features: pd.DataFrame, targets: pd.Series, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[tuple[np.ndarray, np.ndarray]]]:
    """Create a stratified holdout and outer folds from the remaining training set."""
    features, targets = features.align(targets, join="inner", axis=0)
    positions = np.arange(len(features))
    train_positions, test_positions = train_test_split(
        positions,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=targets.to_numpy(),
    )
    y_train = targets.iloc[train_positions]
    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() < N_SPLITS:
        raise ValueError(
            f"Target {targets.name!r} does not have at least {N_SPLITS} examples "
            "of each class in the training split; reduce --test-size or N_SPLITS."
        )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = list(splitter.split(features.iloc[train_positions], y_train))
    return (
        features.iloc[train_positions],
        features.iloc[test_positions],
        y_train,
        targets.iloc[test_positions],
        folds,
    )


def _save_split(
    directory: str,
    features: pd.DataFrame,
    targets: pd.Series,
    filename: str,
) -> None:
    os.makedirs(directory, exist_ok=True)
    data = features.copy()
    data.insert(0, "sample_id", data.index)
    data["target"] = targets.reindex(features.index).to_numpy()
    data.to_csv(os.path.join(directory, filename), index=False)


def _load_saved_holdout(directory: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the persisted holdout split used by ``--load-artifacts``."""
    train = pd.read_csv(os.path.join(directory, "holdout_train.csv"), index_col="sample_id")
    test = pd.read_csv(os.path.join(directory, "holdout_test.csv"), index_col="sample_id")
    train_targets = train.pop("target").astype(int)
    test_targets = test.pop("target").astype(int)
    return train, test, train_targets, test_targets


def _load_saved_fold_tests(
    directory: str, feature_columns: pd.Index, n_folds: int
) -> list[tuple[pd.DataFrame, pd.Series]]:
    """Load the persisted validation partition for each training fold."""
    fold_tests = []
    for fold_number in range(1, n_folds + 1):
        frame = pd.read_csv(
            os.path.join(directory, f"fold_{fold_number:02d}_test.csv"),
            index_col="sample_id",
        )
        target = frame.pop("target").astype(int)
        fold_tests.append((frame.reindex(columns=feature_columns), target))
    return fold_tests


def _make_estimator(estimator, param_grid: dict):
    if not param_grid:
        return clone(estimator)
    return GridSearchCV(
        estimator,
        param_grid,
        scoring=SCORERS["f1"],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
    )


def _metric_values(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def _write_training_summary(
    artifact_dir: str,
    target_name: str,
    model_name: str,
    representation: str,
    summary: dict,
    confusion: np.ndarray,
) -> tuple[str, str]:
    """Save a JSON report and a confusion-matrix plot for a fitted model."""
    key = "__".join(map(_safe_name, (target_name, model_name, representation)))
    summary_dir = os.path.join(artifact_dir, "summaries")
    plot_dir = os.path.join(artifact_dir, "confusion_matrices")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{key}.json")
    plot_path = os.path.join(plot_dir, f"{key}.png")

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["No hint (0)", "Hint (1)"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set_title(f"Holdout confusion matrix — {target_name}\n{model_name} / {representation}")
    display.figure_.tight_layout()
    display.figure_.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)

    summary["confusion_matrix_plot"] = plot_path
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False, default=str)
    return summary_path, plot_path


def _evaluate_model(
    model_name: str,
    representation: str,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series,
    test_targets: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    estimator,
    param_grid: dict,
    target_name: str,
    artifact_dir: str,
    *,
    load_artifacts: bool = False,
) -> dict:
    model_path, _ = _artifact_paths(
        artifact_dir, target_name, model_name, representation
    )
    _, split_dir = _artifact_paths(artifact_dir, target_name, model_name, representation)
    if load_artifacts:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Saved model artifact not found: {model_path}. Run once without --load-artifacts."
            )
        artifact = joblib.load(model_path)
        if artifact.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"Unsupported artifact version in {model_path}")
        fold_models = artifact["fold_models"]
        final_model = artifact["final_model"]
        fold_test_data = _load_saved_fold_tests(
            split_dir, train_features.columns, len(fold_models)
        )
    else:
        fold_models = []
        fold_test_data = []
        fold_scores = []
        fold_importances = []
        best_params = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            for train_indices, validation_indices in folds:
                fold_estimator = _make_estimator(estimator, param_grid)
                X_fold_train = train_features.iloc[train_indices]
                y_fold_train = train_targets.iloc[train_indices]
                X_fold_test = train_features.iloc[validation_indices]
                y_fold_test = train_targets.iloc[validation_indices]
                fold_estimator.fit(X_fold_train, y_fold_train)
                fold_models.append(fold_estimator)
                fold_test_data.append((X_fold_test, y_fold_test))
                fold_scores.append(_metric_values(y_fold_test, fold_estimator.predict(X_fold_test)))
                fitted = (
                    fold_estimator.best_estimator_
                    if isinstance(fold_estimator, GridSearchCV)
                    else fold_estimator
                )
                fold_importances.append(fitted.feature_importances_)
                best_params.append(
                    fold_estimator.best_params_
                    if isinstance(fold_estimator, GridSearchCV)
                    else {}
                )

            final_model = _make_estimator(estimator, param_grid)
            final_model.fit(train_features, train_targets)
            holdout_data = (test_features, test_targets)
            holdout_scores = _metric_values(
                test_targets, final_model.predict(test_features)
            )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "target": target_name,
                "model": model_name,
                "representation": representation,
                "fold_models": fold_models,
                "final_model": final_model,
            },
            model_path,
        )

    if load_artifacts:
        fold_scores = [
            _metric_values(y_true, fitted.predict(X_test))
            for fitted, (X_test, y_true) in zip(fold_models, fold_test_data)
        ]
        fitted_final = final_model
        holdout_scores = _metric_values(
            test_targets, fitted_final.predict(test_features)
        )
        fold_importances = [
            (fitted.best_estimator_ if isinstance(fitted, GridSearchCV) else fitted)
            .feature_importances_
            for fitted in fold_models
        ]
        best_params = [
            fitted.best_params_ if isinstance(fitted, GridSearchCV) else {}
            for fitted in fold_models
        ]

    importance = pd.DataFrame(fold_importances, columns=train_features.columns).mean(axis=0)
    holdout_predictions = final_model.predict(test_features)
    holdout_scores = _metric_values(test_targets, holdout_predictions)
    confusion = confusion_matrix(test_targets, holdout_predictions, labels=[0, 1])
    fold_details = [
        {
            "fold": fold_number,
            "validation_samples": len(y_true),
            "metrics": score,
            "best_params": best_params[fold_number - 1],
        }
        for fold_number, ((_, y_true), score) in enumerate(
            zip(fold_test_data, fold_scores), start=1
        )
    ]
    final_estimator = (
        final_model.best_estimator_
        if isinstance(final_model, GridSearchCV)
        else final_model
    )
    final_feature_importance = getattr(final_estimator, "feature_importances_", None)
    model_summary = {
        "artifact_version": ARTIFACT_VERSION,
        "mission": MISSION,
        "target": target_name,
        "model": model_name,
        "representation": representation,
        "model_class": type(final_estimator).__name__,
        "model_path": model_path,
        "split_directory": split_dir,
        "total_samples": len(train_targets) + len(test_targets),
        "total_class_counts": {
            str(label): int(count)
            for label, count in pd.concat([train_targets, test_targets])
            .value_counts()
            .sort_index()
            .items()
        },
        "training_samples": len(train_targets),
        "holdout_samples": len(test_targets),
        "training_class_counts": {
            str(label): int(count)
            for label, count in train_targets.value_counts().sort_index().items()
        },
        "holdout_class_counts": {
            str(label): int(count)
            for label, count in test_targets.value_counts().sort_index().items()
        },
        "n_features": train_features.shape[1],
        "feature_names": list(train_features.columns),
        "random_state": RANDOM_STATE,
        "cv_folds": len(fold_models),
        "inner_search_folds": 3 if param_grid else None,
        "parameter_grid": param_grid,
        "best_params_per_fold": best_params,
        "final_best_params": (
            final_model.best_params_ if isinstance(final_model, GridSearchCV) else {}
        ),
        "final_model_parameters": final_estimator.get_params(deep=False),
        "cv_fold_results": fold_details,
        "cv_metrics_mean": {
            metric: float(np.mean([score[metric] for score in fold_scores]))
            for metric in SCORERS
        },
        "cv_metrics_std": {
            metric: float(np.std([score[metric] for score in fold_scores]))
            for metric in SCORERS
        },
        "holdout_metrics": holdout_scores,
        "holdout_confusion_matrix": {
            "labels": [0, 1],
            "rows": "actual",
            "columns": "predicted",
            "values": confusion.tolist(),
        },
        "final_feature_importance": (
            dict(zip(train_features.columns, final_feature_importance.tolist()))
            if final_feature_importance is not None
            else {}
        ),
    }
    summary_path, plot_path = _write_training_summary(
        artifact_dir,
        target_name,
        model_name,
        representation,
        model_summary,
        confusion,
    )
    return {
        "model": model_name,
        "representation": representation,
        "target": target_name,
        "n_features": train_features.shape[1],
        "feature_importance": importance,
        "best_params_per_fold": best_params,
        **{f"cv_{metric}": np.mean([score[metric] for score in fold_scores]) for metric in SCORERS},
        **{
            f"cv_{metric}_std": np.std([score[metric] for score in fold_scores])
            for metric in SCORERS
        },
        **{f"test_{metric}": value for metric, value in holdout_scores.items()},
        "model_path": model_path,
        "summary_path": summary_path,
        "confusion_matrix_plot": plot_path,
    }


class HintExperiment:
    """Przebieg eksperymentu klasyfikacji podpowiedzi; parametry odpowiadają opcjom CLI ``main``."""

    def __init__(
        self,
        target_set: Literal["all", "hints", "any"] = "all",
        include_elapsed_seconds: bool = False,
        quick: bool = False,
        load_artifacts: bool = False,
        test_size: float = TEST_SIZE,
        artifacts_dir: str | None = None,
    ) -> None:
        if target_set not in ("all", "hints", "any"):
            raise ValueError("target_set must be 'all', 'hints' or 'any'")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        self.target_set: str = target_set
        self.include_elapsed_seconds: bool = include_elapsed_seconds
        self.quick: bool = quick
        self.load_artifacts: bool = load_artifacts
        self.test_size: float = test_size
        self.output_suffix: str = "_with_elapsed_seconds" if include_elapsed_seconds else ""
        self.artifacts_dir: str = artifacts_dir or os.path.join(
            FEATURES_DIR,
            "experiment_artifacts",
            f"{MISSION}{self.output_suffix}_test{test_size:g}",
        )

    def _models(self) -> list[tuple[str, object, dict]]:
        models = [
            ("baseline_xgb", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE
            ), {}),
            ("baseline_decision_tree", DecisionTreeClassifier(random_state=RANDOM_STATE), {}),
            (
                "xgboost",
                XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss", random_state=RANDOM_STATE
                ),
                XGBOOST_PARAM_GRID,
            ),
            (
                "decision_tree",
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                DECISION_TREE_PARAM_GRID,
            ),
        ]
        return models[:2] if self.quick else models

    def run(self) -> pd.DataFrame:
        """Trenuje lub wczytuje modele, zapisuje metryki i ważność cech, zwraca tabelę metryk."""
        xsml_features, xsml_targets = _load_mission(
            _load_xsml_features, MISSION, include_elapsed_seconds=self.include_elapsed_seconds
        )
        flat_features, _ = _load_mission(
            _load_flat_features, MISSION, include_elapsed_seconds=self.include_elapsed_seconds
        )
        representations = {"xsml": xsml_features, "flat": flat_features}
        target_names = list(xsml_targets.columns)
        if self.target_set == "hints":
            target_names.remove("any_hint")
        elif self.target_set == "any":
            target_names = ["any_hint"]

        results = []
        for target_name in target_names:
            target = xsml_targets[target_name]
            for representation, features in representations.items():
                _, split_dir = _artifact_paths(self.artifacts_dir, target_name, "", representation)
                if self.load_artifacts:
                    train_features, test_features, train_targets, test_targets = (
                        _load_saved_holdout(split_dir)
                    )
                    folds = []
                else:
                    train_features, test_features, train_targets, test_targets, folds = (
                        _make_splits(features, target, self.test_size)
                    )
                    _save_split(split_dir, train_features, train_targets, "holdout_train.csv")
                    _save_split(split_dir, test_features, test_targets, "holdout_test.csv")
                    for number, (train_idx, validation_idx) in enumerate(folds, 1):
                        _save_split(
                            split_dir,
                            train_features.iloc[train_idx],
                            train_targets.iloc[train_idx],
                            f"fold_{number:02d}_train.csv",
                        )
                        _save_split(
                            split_dir,
                            train_features.iloc[validation_idx],
                            train_targets.iloc[validation_idx],
                            f"fold_{number:02d}_test.csv",
                        )

                for model_name, estimator, param_grid in self._models():
                    results.append(
                        _evaluate_model(
                            model_name,
                            representation,
                            train_features,
                            test_features,
                            train_targets,
                            test_targets,
                            folds,
                            estimator,
                            param_grid,
                            target_name,
                            self.artifacts_dir,
                            load_artifacts=self.load_artifacts,
                        )
                    )

        importance_matrix = pd.concat(
            {
                f"{result['target']}|{result['model']}|{result['representation']}": result.pop("feature_importance")
                for result in results
            },
            axis=1,
        ).fillna(0.0)
        importance_matrix.index.name = "feature"
        importance_matrix.to_csv(
            os.path.join(FEATURES_DIR, f"{MISSION}{self.output_suffix}_feature_importance.csv")
        )
        summary = pd.DataFrame(results).set_index(["target", "model", "representation"])
        summary.to_csv(
            os.path.join(FEATURES_DIR, f"{MISSION}{self.output_suffix}_classification_metrics.csv")
        )
        return summary


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--target-set",
        choices=("all", "hints", "any"),
        default="all",
        help="Evaluate every hint and their OR (all), hints only, or the OR label only.",
    )
    argument_parser.add_argument(
        "--include-elapsed-seconds",
        action="store_true",
        help="Keep elapsed_seconds as a predictor and train on each snapshot without collapsing states.",
    )
    argument_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only baseline XGBoost and decision tree, without hyperparameter search.",
    )
    argument_parser.add_argument(
        "--load-artifacts",
        action="store_true",
        help="Recalculate metrics from saved splits and fitted models without training.",
    )
    argument_parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of samples reserved for final holdout validation (default: 0.2).",
    )
    argument_parser.add_argument(
        "--artifacts-dir",
        help="Artifact directory (default: data/features/experiment_artifacts/<mission>...).",
    )
    args = argument_parser.parse_args()
    if not 0 < args.test_size < 1:
        argument_parser.error("--test-size must be between 0 and 1")

    output_suffix = "_with_elapsed_seconds" if args.include_elapsed_seconds else ""
    test_suffix = f"_test{args.test_size:g}"
    artifact_dir = args.artifacts_dir or os.path.join(
        FEATURES_DIR, "experiment_artifacts", f"{MISSION}{output_suffix}{test_suffix}"
    )

    print("Selected target set:", args.target_set)
    print("Include elapsed_seconds:", args.include_elapsed_seconds)
    print("Load saved artifacts:", args.load_artifacts)
    print("Artifact directory:", artifact_dir)

    xsml_features, xsml_targets = _load_mission(
        _load_xsml_features, MISSION, include_elapsed_seconds=args.include_elapsed_seconds
    )
    flat_features, flat_targets = _load_mission(
        _load_flat_features, MISSION, include_elapsed_seconds=args.include_elapsed_seconds
    )
    models = [
        ("baseline_xgb", XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ), {}),
        ("baseline_decision_tree", DecisionTreeClassifier(random_state=RANDOM_STATE), {}),
        (
            "xgboost",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            XGBOOST_PARAM_GRID,
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            DECISION_TREE_PARAM_GRID,
        ),
    ]
    if args.quick:
        models = models[:2]
    representations = {
        "xsml": xsml_features,
        "flat": flat_features,
    }
    target_names = list(xsml_targets.columns)
    if args.target_set == "hints":
        target_names.remove("any_hint")
    elif args.target_set == "any":
        target_names = ["any_hint"]

    results = []
    for target_name in target_names:
        target = xsml_targets[target_name]
        for representation, features in representations.items():
            _, split_dir = _artifact_paths(
                artifact_dir, target_name, "", representation
            )
            if args.load_artifacts:
                train_features, test_features, train_targets, test_targets = (
                    _load_saved_holdout(split_dir)
                )
                folds = []
            else:
                train_features, test_features, train_targets, test_targets, folds = (
                    _make_splits(features, target, args.test_size)
                )
                _save_split(split_dir, train_features, train_targets, "holdout_train.csv")
                _save_split(split_dir, test_features, test_targets, "holdout_test.csv")
                for fold_number, (train_indices, validation_indices) in enumerate(folds, 1):
                    _save_split(
                        split_dir,
                        train_features.iloc[train_indices],
                        train_targets.iloc[train_indices],
                        f"fold_{fold_number:02d}_train.csv",
                    )
                    _save_split(
                        split_dir,
                        train_features.iloc[validation_indices],
                        train_targets.iloc[validation_indices],
                        f"fold_{fold_number:02d}_test.csv",
                    )

            for model_name, estimator, param_grid in models:
                results.append(
                    _evaluate_model(
                        model_name,
                        representation,
                        train_features,
                        test_features,
                        train_targets,
                        test_targets,
                        folds,
                        estimator,
                        param_grid,
                        target_name,
                        artifact_dir,
                        load_artifacts=args.load_artifacts,
                    )
                )

    importance_matrix = pd.concat(
        {
            f"{result['target']}|{result['model']}|{result['representation']}": result.pop("feature_importance")
            for result in results
        },
        axis=1,
    ).fillna(0.0)
    importance_matrix.index.name = "feature"
    importance_path = os.path.join(
        FEATURES_DIR, f"{MISSION}{output_suffix}_feature_importance.csv"
    )
    metrics_path = os.path.join(
        FEATURES_DIR, f"{MISSION}{output_suffix}_classification_metrics.csv"
    )
    importance_matrix.to_csv(importance_path)
    summary = pd.DataFrame(results).set_index(["target", "model", "representation"])
    summary.to_csv(metrics_path)

    pd.set_option("display.width", 120)
    row_description = (
        "snapshot rows"
        if args.include_elapsed_seconds
        else "state rows after idle-time collapse"
    )
    print(f"Mission: {MISSION} | {row_description}: {len(xsml_features)}")
    print(f"Targets evaluated: {', '.join(target_names)}")
    print(f"\nStratified {N_SPLITS}-fold CV on hardening (unseen states from this mission):")
    print(summary[[
        "n_features", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1",
        "test_accuracy", "test_precision", "test_recall", "test_f1",
    ]])
    print("\nMean feature-importance matrix (rows=features, columns=target|model|representation):")
    print(importance_matrix.sort_index().round(4).to_string())
    print(f"\nFeature-importance matrix saved to: {importance_path}")
    print(f"Classification metrics saved to: {metrics_path}")
    print(f"Splits and fitted models saved in: {artifact_dir}")


if __name__ == "__main__":
    main()
