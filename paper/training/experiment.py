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

Nested stratified 10-fold CV evaluates generalization to unseen states from this
mission: each outer training fold gets a separate 3-fold hyperparameter search.
Leave-one-mission-out validation is not applicable to a single mission.
"""

from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MISSION = "hardening_linux"
N_SPLITS = 5
RANDOM_STATE = 42

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


def _evaluate_model(
    model_name: str,
    representation: str,
    features: pd.DataFrame,
    targets: pd.Series,
    estimator,
    param_grid: dict,
    target_name: str,
) -> dict:
    features, targets = features.align(targets, join="inner", axis=0)
    stratified_cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    model = (
        GridSearchCV(
            estimator,
            param_grid,
            scoring=SCORERS["f1"],
            cv=inner_cv,
            refit=True,
        )
        if param_grid
        else estimator
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        cv_scores = cross_validate(
            model,
            features,
            targets,
            cv=stratified_cv,
            scoring=SCORERS,
            return_estimator=True,
        )

    fold_importances = pd.DataFrame(
        [
            (
                fitted.best_estimator_.feature_importances_
                if isinstance(fitted, GridSearchCV)
                else fitted.feature_importances_
            )
            for fitted in cv_scores["estimator"]
        ],
        columns=features.columns,
    )

    return {
        "model": model_name,
        "representation": representation,
        "target": target_name,
        "n_features": features.shape[1],
        "feature_importance": fold_importances.mean(axis=0),
        "best_params_per_fold": [
            fitted.best_params_ if isinstance(fitted, GridSearchCV) else {}
            for fitted in cv_scores["estimator"]
        ],
        "cv_accuracy": cv_scores["test_accuracy"].mean(),
        "cv_precision": cv_scores["test_precision"].mean(),
        "cv_recall": cv_scores["test_recall"].mean(),
        "cv_f1": cv_scores["test_f1"].mean(),
    }


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
    args = argument_parser.parse_args()

    print("Selected target set:", args.target_set)
    print("Include elapsed_seconds:", args.include_elapsed_seconds)

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
        for model_name, estimator, param_grid in models:
            for representation, features in representations.items():
                results.append(
                    _evaluate_model(
                        model_name,
                        representation,
                        features,
                        target,
                        estimator,
                        param_grid,
                        target_name,
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
    output_suffix = "_with_elapsed_seconds" if args.include_elapsed_seconds else ""
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
    print(summary[["n_features", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1"]])
    print("\nMean feature-importance matrix (rows=features, columns=target|model|representation):")
    print(importance_matrix.sort_index().round(4).to_string())
    print(f"\nFeature-importance matrix saved to: {importance_path}")
    print(f"Classification metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
