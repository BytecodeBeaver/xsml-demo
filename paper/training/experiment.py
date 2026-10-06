"""Compare XGBoost and decision-tree classifiers on the hardening mission.

The multi-label targets are collapsed to one binary "any hint shown" label.
Rows with the same state but different idle-time snapshots are collapsed into
one state row, positive if any snapshot showed a hint. XSML features include a
16-value goal-graph embedding; both representations include rule-derived
action and goal predicates.

``elapsed_seconds`` is dropped from the predictors: it trivially predicts
hints (the support rules key off the same idle-time threshold),
which masks whether the state-describing columns themselves are useful.
Without it, several idle-time snapshots of the same state become identical
rows, so they are collapsed into one row per state; the collapsed target is
True if any of its snapshots would have shown a hint.

Nested stratified 5-fold CV evaluates generalization to unseen states from this
mission: each outer training fold gets a separate 3-fold hyperparameter search.
Leave-one-mission-out validation is not applicable to a single mission.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")
MISSION = "hardening_linux"
N_SPLITS = 5
RANDOM_STATE = 2142135243

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


def _load_targets(mission: str) -> pd.Series:
    """Collapse the multi-label hint targets into one mission-agnostic "any hint shown" label."""
    targets = pd.read_csv(
        os.path.join(FEATURES_DIR, f"{mission}_targets.csv"),
        true_values=["true"],
        false_values=["false"],
    )
    hint_columns = [column for column in targets.columns if column.startswith("hint::")]
    any_hint = targets[hint_columns].any(axis=1).astype(int)
    return pd.Series(any_hint.values, index=targets["sample_id"], name="any_hint")


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


def _load_mission(loader, mission: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a mission and collapse idle-time copies to one row per state valuation."""
    features, targets = loader(mission).align(_load_targets(mission), join="inner", axis=0)
    features = features.drop(columns=["elapsed_seconds"])
    feature_columns = list(features.columns)
    merged = features.copy()
    merged["any_hint"] = targets.values
    collapsed = merged.groupby(feature_columns, as_index=False, sort=False)["any_hint"].max()
    return collapsed[feature_columns], collapsed["any_hint"]


def _evaluate_model(
    model_name: str,
    representation: str,
    features: pd.DataFrame,
    targets: pd.Series,
    estimator,
    param_grid: dict,
) -> dict:
    features, targets = features.align(targets, join="inner", axis=0)
    stratified_cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator,
        param_grid,
        scoring=SCORERS["f1"],
        cv=inner_cv,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        cv_scores = cross_validate(
            search,
            features,
            targets,
            cv=stratified_cv,
            scoring=SCORERS,
            return_estimator=True,
        )

    return {
        "model": model_name,
        "representation": representation,
        "n_features": features.shape[1],
        "best_params_per_fold": [fitted.best_params_ for fitted in cv_scores["estimator"]],
        "cv_accuracy": cv_scores["test_accuracy"].mean(),
        "cv_precision": cv_scores["test_precision"].mean(),
        "cv_recall": cv_scores["test_recall"].mean(),
        "cv_f1": cv_scores["test_f1"].mean(),
    }


def main() -> None:
    xsml_features, xsml_targets = _load_mission(_load_xsml_features, MISSION)
    flat_features, flat_targets = _load_mission(_load_flat_features, MISSION)
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
    representations = {
        "xsml": xsml_features,
        "flat": flat_features,
    }
    results = [
        _evaluate_model(model_name, representation, features, xsml_targets, estimator, param_grid)
        for model_name, estimator, param_grid in models
        for representation, features in representations.items()
    ]

    pd.set_option("display.width", 120)
    counts = xsml_targets.value_counts().sort_index()
    print(f"Mission: {MISSION} | state samples after idle-time collapse: {len(xsml_targets)}")
    print(f"Negative: {int(counts.get(0, 0))} | positive: {int(counts.get(1, 0))}")

    summary = pd.DataFrame(results).set_index(["model", "representation"])
    print(f"\nStratified {N_SPLITS}-fold CV on hardening (unseen states from this mission):")
    print(summary[["n_features", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1"]])


if __name__ == "__main__":
    main()
