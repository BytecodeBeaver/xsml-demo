from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.parser import SimulationMission, XSMLParser


@dataclass(frozen=True)
class FeatureMatrices:
    """Macierze cech i zmiennych docelowych dla tych samych migawek stanu."""

    xsml: pd.DataFrame
    flat: pd.DataFrame
    targets: pd.DataFrame
    flat_records: pd.DataFrame


def _json_value(value: Any) -> str:
    """Serializuje wartości skalarne YAML w spójny sposób na potrzeby nazw cech i plików CSV."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _feature_value_label(value: Any) -> str:
    """Formatuje czytelną etykietę kategorii bez opakowania JSON ani prefiksów."""
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return _json_value(value)


def _requirements_match(requirements: dict, inventory: dict, worldstate: dict) -> bool:
    """Stosuje semantykę GameLogic, także dla nieobecnych zmiennych wymaganych jako false."""
    for name, expected in requirements.get("inventory", {}).items():
        if name == "player_idle_time":
            continue
        actual = inventory.get(name)
        if actual is None and expected is False:
            continue
        if actual != expected:
            return False
    for name, expected in requirements.get("worldstate", {}).items():
        actual = worldstate.get(name)
        if actual is None and expected is False:
            continue
        if actual != expected:
            return False
    return True


def _walk_goals(goals: list[dict]) -> Iterable[dict]:
    for goal in goals or []:
        yield goal
        yield from _walk_goals(goal.get("subgoals", []))


class XSMLFeatureExtractor:
    """Generuje macierze stanu dla wszystkich wartości boolowskich zadeklarowanych w XSML.

    Każda możliwa wartość ekwipunku lub stanu świata staje się cechą boolowską. Uwzględniane
    są także predykaty działań i celów, upływający czas bezczynności oraz reprezentacja
    hierarchii celów.
    """

    def __init__(
        self,
        mission: SimulationMission,
        *,
        max_states: int = 10_000,
        inventory_noise_variants: int = 2,
        random_seed: int | None = 42,
    ):
        self.mission = mission
        self.max_states = max_states
        if inventory_noise_variants < 0:
            raise ValueError("Liczba wariantów szumu inventory nie może być ujemna.")
        self.inventory_noise_variants = inventory_noise_variants
        self.random = random.Random(random_seed)
        self.actions = list(mission.actions or [])
        self.goals = list(_walk_goals(mission.goals or []))
        self.hints = list(mission.support or [])
        self.inventory_names, self.worldstate_names = self._collect_state_variables()
        self.inventory_domains, self.worldstate_domains = self._collect_state_domains()
        self.idle_times = self._collect_idle_times()
        self.goal_hierarchy = self._describe_goal_hierarchy()

    def _collect_state_variables(self) -> tuple[list[str], list[str]]:
        inventory_names = set((self.mission.inventory or {}).keys())
        worldstate_names = set((self.mission.worldstate or {}).keys())

        def add_requirements(requirements: dict) -> None:
            inventory_names.update(requirements.get("inventory", {}).keys() - {"player_idle_time"})
            worldstate_names.update(requirements.get("worldstate", {}).keys())

        for goal in self.goals:
            add_requirements(goal.get("requires") or {})
        for action in self.actions:
            add_requirements(action.get("requires") or {})
            for produced in action.get("produces") or []:
                change = produced.get("change") or {}
                inventory_names.update((change.get("inventory") or {}).keys())
                worldstate_names.update((change.get("worldstate") or {}).keys())
                shared = (change.get("inventory_and_worldstate") or {}).keys()
                inventory_names.update(shared)
                worldstate_names.update(shared)
        for hint in self.hints:
            add_requirements(hint.get("requires") or {})
        return sorted(inventory_names), sorted(worldstate_names)

    def _collect_state_domains(self) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Zbiera wartości zadeklarowane w XSML, rozszerzając fakty boolowskie o obie wartości."""
        values_by_scope: dict[str, dict[str, list[Any]]] = {
            "inventory": {name: [] for name in self.inventory_names},
            "worldstate": {name: [] for name in self.worldstate_names},
        }

        def add_values(scope: str, values: dict) -> None:
            for name, value in values.items():
                if name == "player_idle_time":
                    continue
                candidates = values_by_scope[scope].setdefault(name, [])
                encoded = _json_value(value)
                if all(_json_value(existing) != encoded for existing in candidates):
                    candidates.append(value)

        add_values("inventory", self.mission.inventory or {})
        add_values("worldstate", self.mission.worldstate or {})
        requirements = [goal.get("requires") or {} for goal in self.goals]
        requirements.extend(action.get("requires") or {} for action in self.actions)
        requirements.extend(hint.get("requires") or {} for hint in self.hints)
        for requirement in requirements:
            for scope in ("inventory", "worldstate"):
                add_values(scope, requirement.get(scope) or {})

        for action in self.actions:
            for outcome in action.get("produces") or []:
                change = outcome.get("change") or {}
                for scope in ("inventory", "worldstate"):
                    add_values(scope, change.get(scope) or {})
                shared = change.get("inventory_and_worldstate") or {}
                add_values("inventory", shared)
                add_values("worldstate", shared)

        domains: dict[str, dict[str, list[Any]]] = {"inventory": {}, "worldstate": {}}
        for scope, variables in values_by_scope.items():
            for name, candidates in variables.items():
                if not candidates or all(type(value) is bool for value in candidates):
                    domains[scope][name] = [False, True]
                else:
                    domains[scope][name] = sorted(candidates, key=_json_value)
        return domains["inventory"], domains["worldstate"]

    def _describe_goal_hierarchy(self) -> dict[str, dict[str, Any]]:
        """Zwraca jawne metadane strukturalne dla każdego celu."""
        hierarchy: dict[str, dict[str, Any]] = {}

        def visit(goal: dict, depth: int, sibling_count: int, parent_id: str | None) -> None:
            goal_id = goal["id"]
            children = goal.get("subgoals", [])
            hierarchy[goal_id] = {
                "goal_depth": depth,
                "number_of_children": len(children),
                "is_leaf": not children,
                "is_root": parent_id is None,
                "number_of_ancestors": depth,
                "number_of_siblings": sibling_count,
                "parent_id": parent_id,
            }
            for child in children:
                visit(child, depth + 1, len(children) - 1, goal_id)

        roots = self.mission.goals or []
        for goal in roots:
            visit(goal, 0, len(roots) - 1, None)
        return hierarchy

    def _collect_idle_times(self) -> list[float]:
        thresholds = {
            float((hint.get("requires") or {}).get("inventory", {}).get("player_idle_time"))
            for hint in self.hints
            if (hint.get("requires") or {}).get("inventory", {}).get("player_idle_time") is not None
        }
        if not thresholds:
            return [0.0]
        times = {0.0}
        for threshold in thresholds:
            times.add(max(0.0, threshold - 1.0))
            times.add(threshold)
        return sorted(times)

    def _randomized_elapsed_seconds(self, scheduled_time: float) -> float:
        """Losuje wartość po tej samej stronie progu 15 sekund co zaplanowany czas."""
        unit_interval = self.random.random()
        if scheduled_time < 15.0:
            return unit_interval * 15.0
        return 15.0 + unit_interval * 15.0

    def _noisy_inventory_variants(self, inventory: dict) -> list[dict]:
        """Zwraca różne, powtarzalne warianty alternatywnego obrazu ekwipunku gracza."""
        if not self.inventory_noise_variants or not self.inventory_names:
            return []

        alternatives = []
        domains = [self.inventory_domains[name] for name in self.inventory_names]
        for values in product(*domains):
            variant = dict(zip(self.inventory_names, values))
            if variant == inventory:
                continue
            alternatives.append(variant)
        count = min(self.inventory_noise_variants, len(alternatives))
        return self.random.sample(alternatives, count)

    def _outcomes(self, action: dict) -> Iterable[dict]:
        for produced in action.get("produces") or []:
            if float(produced.get("probability", 1.0)) > 0:
                yield produced

    @staticmethod
    def _apply_outcome(inventory: dict, worldstate: dict, produced: dict) -> tuple[dict, dict]:
        next_inventory = dict(inventory)
        next_worldstate = dict(worldstate)
        change = produced.get("change") or {}
        next_inventory.update(change.get("inventory") or {})
        next_worldstate.update(change.get("worldstate") or {})
        shared = change.get("inventory_and_worldstate") or {}
        next_inventory.update(shared)
        next_worldstate.update(shared)
        return next_inventory, next_worldstate

    def _enumerate_states(self) -> list[tuple[dict, dict]]:
        variables = [
            *(("inventory", name, self.inventory_domains[name]) for name in self.inventory_names),
            *(("worldstate", name, self.worldstate_domains[name]) for name in self.worldstate_names),
        ]
        state_count = math.prod(len(domain) for _, _, domain in variables)
        if state_count > self.max_states:
            raise ValueError(
                f"Warianty wartości wymagają {state_count} stanów, limit wynosi {self.max_states}."
            )

        states = []
        for values in product(*(domain for _, _, domain in variables)):
            inventory: dict[str, Any] = {}
            worldstate: dict[str, Any] = {}
            for (scope, name, _), value in zip(variables, values):
                (inventory if scope == "inventory" else worldstate)[name] = value
            states.append((inventory, worldstate))
        return states

    def _goal_values(self, inventory: dict, worldstate: dict) -> dict[str, bool]:
        values: dict[str, bool] = {}

        def evaluate(goal: dict) -> bool:
            subgoal_results = [evaluate(child) for child in goal.get("subgoals", [])]
            subgoals_met = all(subgoal_results)
            achieved = _requirements_match(goal.get("requires") or {}, inventory, worldstate) and subgoals_met
            values[goal["id"]] = achieved
            return achieved

        for goal in self.mission.goals or []:
            evaluate(goal)
        return values

    def _available_hints(self, inventory: dict, worldstate: dict, elapsed: float) -> set[str]:
        available = set()
        for hint in self.hints:
            requirements = hint.get("requires") or {}
            if not _requirements_match(requirements, inventory, worldstate):
                continue
            threshold = requirements.get("inventory", {}).get("player_idle_time")
            if threshold is None or elapsed >= float(threshold):
                available.add(hint["id"])
        return available

    def _flat_record(self, inventory: dict, worldstate: dict, elapsed: float, sample_id: str) -> dict:
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "mission_id": self.mission.id,
            "elapsed_seconds": elapsed,
        }
        for name in self.inventory_names:
            record[f"inventory.{name}"] = inventory.get(name)
        for name in self.worldstate_names:
            record[f"worldstate.{name}"] = worldstate.get(name)
        for action in self.actions:
            record[f"action_available.{action['id']}"] = _requirements_match(
                action.get("requires") or {}, inventory, worldstate
            )
        for goal_id, completed in sorted(self._goal_values(inventory, worldstate).items()):
            record[f"goal_completed.{goal_id}"] = completed
        return record

    def _xsml_features(self, record: dict) -> dict:
        features: dict[str, Any] = {"elapsed_seconds": record["elapsed_seconds"]}
        for scope, names in (("inventory", self.inventory_names), ("worldstate", self.worldstate_names)):
            for name in names:
                value = record[f"{scope}.{name}"]
                domains = self.inventory_domains if scope == "inventory" else self.worldstate_domains
                values = domains[name]
                feature_name = self._state_feature_name(scope, name)
                if values and all(type(candidate) is bool for candidate in values):
                    features[feature_name] = value
                else:
                    for candidate in values:
                        column = f"{feature_name}=={_feature_value_label(candidate)}"
                        features[column] = value == candidate
        for action in self.actions:
            features[f"action_available.{action['id']}"] = record[f"action_available.{action['id']}"]
        goal_values = self._goal_values_from_record(record)
        for goal_id, metadata in self.goal_hierarchy.items():
            features[f"goal_completed.{goal_id}"] = goal_values[goal_id]
            for key in (
                "goal_depth",
                "number_of_children",
                "is_leaf",
                "is_root",
                "number_of_ancestors",
                "number_of_siblings",
            ):
                features[f"{key}.{goal_id}"] = metadata[key]
            parent_id = metadata["parent_id"]
            features[f"parent_completed.{goal_id}"] = (
                goal_values[parent_id] if parent_id is not None else False
            )
        return features

    def _goal_values_from_record(self, record: dict) -> dict[str, bool]:
        return {
            goal_id: bool(record[f"goal_completed.{goal_id}"])
            for goal_id in self.goal_hierarchy
        }

    def _state_feature_name(self, scope: str, name: str) -> str:
        """Używa tej samej nazwy stanu z kwalifikatorem zakresu w obu macierzach cech."""
        return f"{scope}.{name}"

    def _column_schemas(self) -> tuple[list[str], list[str]]:
        fact_columns = []
        for scope, names in (("inventory", self.inventory_names), ("worldstate", self.worldstate_names)):
            domains = self.inventory_domains if scope == "inventory" else self.worldstate_domains
            for name in names:
                values = domains[name]
                feature_name = self._state_feature_name(scope, name)
                if values and all(type(value) is bool for value in values):
                    fact_columns.append(feature_name)
                else:
                    fact_columns.extend(
                        f"{feature_name}=={_feature_value_label(value)}" for value in values
                    )
        feature_columns = ["elapsed_seconds", *fact_columns]
        feature_columns.extend(f"action_available.{action['id']}" for action in self.actions)
        for goal_id in sorted(self.goal_hierarchy):
            feature_columns.append(f"goal_completed.{goal_id}")
            feature_columns.extend(
                f"{key}.{goal_id}"
                for key in (
                    "goal_depth",
                    "number_of_children",
                    "is_leaf",
                    "is_root",
                    "number_of_ancestors",
                    "number_of_siblings",
                    "parent_completed",
                )
            )
        return feature_columns, [f"hint::{hint['id']}" for hint in sorted(self.hints, key=lambda item: item["id"])]

    def extract_training_data(self) -> FeatureMatrices:
        """Zwraca projekcje cech i wieloetykietowe zmienne docelowe dla tej misji.

        ``flat_records`` to tabela migawek bez zmiennych docelowych, gotowa do
        zapisania w CSV. Powiązane macierze są projekcjami tych samych
        wygenerowanych migawek. Zmienne docelowe pozostają oddzielone, aby
        zapobiec wyciekowi danych.
        """
        raw_records = []
        xsml_rows = []
        target_rows = []
        sample_number = 0
        for inventory, worldstate in self._enumerate_states():
            for scheduled_time in self.idle_times:
                sample_id = f"{self.mission.id}:{sample_number}"
                sample_number += 1
                elapsed = self._randomized_elapsed_seconds(scheduled_time)
                record = self._flat_record(inventory, worldstate, elapsed, sample_id)
                raw_records.append(record)
                xsml_rows.append(self._xsml_features(record))
                available = self._available_hints(inventory, worldstate, elapsed)
                target_rows.append({f"hint::{hint['id']}": hint["id"] in available for hint in self.hints})

                # The inventory shown to the model may be mistaken, while action,
                # goal, and hint values continue to describe the underlying state.
                for variant_index, noisy_inventory in enumerate(
                    self._noisy_inventory_variants(inventory)
                ):
                    noisy_record = dict(record)
                    noisy_record["sample_id"] = f"{sample_id}:noise:{variant_index}"
                    for name, value in noisy_inventory.items():
                        noisy_record[f"inventory.{name}"] = value
                    raw_records.append(noisy_record)
                    xsml_rows.append(self._xsml_features(noisy_record))
                    target_rows.append(
                        {f"hint::{hint['id']}": hint["id"] in available for hint in self.hints}
                    )

        feature_columns, target_columns = self._column_schemas()
        flat_records = pd.DataFrame(raw_records)
        for column in ["sample_id", "mission_id", "elapsed_seconds"]:
            if column not in flat_records:
                flat_records[column] = pd.Series(dtype="object")
        flat_records = flat_records.reindex(columns=self._flat_record_columns())
        xsml = pd.DataFrame(xsml_rows).reindex(columns=feature_columns).fillna(False)
        xsml["elapsed_seconds"] = xsml["elapsed_seconds"].astype(float)
        targets = pd.DataFrame(target_rows).reindex(columns=target_columns, fill_value=False).astype(bool)
        sample_index = pd.Index(flat_records["sample_id"], name="sample_id")
        xsml.index = sample_index
        targets.index = sample_index
        categorical_domains = {
            f"{scope}.{name}": domains[name]
            for scope, names, domains in (
                ("inventory", self.inventory_names, self.inventory_domains),
                ("worldstate", self.worldstate_names, self.worldstate_domains),
            )
            for name in names
            if not all(type(value) is bool for value in domains[name])
        }
        flat = FlatSnapshotFeatureExtractor(
            flat_records, categorical_domains=categorical_domains
        ).extract_features()
        flat = flat.reindex(columns=[column for column in xsml.columns if column in flat.columns])
        flat.index = sample_index
        return FeatureMatrices(
            xsml=xsml,
            flat=flat,
            targets=targets,
            flat_records=flat_records.reset_index(drop=True),
        )

    def _flat_record_columns(self) -> list[str]:
        columns = ["sample_id", "mission_id", "elapsed_seconds"]
        columns.extend(f"inventory.{name}" for name in self.inventory_names)
        columns.extend(f"worldstate.{name}" for name in self.worldstate_names)
        columns.extend(f"action_available.{action['id']}" for action in self.actions)
        columns.extend(f"goal_completed.{goal['id']}" for goal in sorted(self.goals, key=lambda item: item["id"]))
        return columns

    def extract_features(self) -> pd.DataFrame:
        """Zwraca macierz cech XSML w reprezentacji propozycjonalnej (jeden wiersz na migawkę)."""
        return self.extract_training_data().xsml


class FlatSnapshotFeatureExtractor:
    """Konwertuje wewnętrznie wygenerowane stany gry na kolumny cech."""

    METADATA_COLUMNS = {"sample_id", "mission_id"}

    def __init__(
        self,
        records: pd.DataFrame,
        *,
        categorical_domains: dict[str, list[Any]] | None = None,
    ):
        self.records = records
        self.categorical_domains = categorical_domains or {}

    def extract_features(self) -> pd.DataFrame:
        if not isinstance(self.records, pd.DataFrame):
            raise TypeError("Dane CSV muszą być obiektem pandas.DataFrame.")
        features = self.records.drop(columns=list(self.METADATA_COLUMNS), errors="ignore").copy()
        for source_column, values in self.categorical_domains.items():
            if source_column not in features:
                continue
            observed = features.pop(source_column)
            for value in values:
                features[f"{source_column}=={_feature_value_label(value)}"] = observed.eq(value)
        return features


def build_paired_feature_matrices(
    xsml_document: dict,
    *,
    max_states: int = 10_000,
    inventory_noise_variants: int = 2,
    random_seed: int | None = 42,
) -> FeatureMatrices:
    """Buduje powiązane cechy z szumem ekwipunku i konfigurowalnym ziarnem generatora."""
    mission = XSMLParser.parse(copy.deepcopy(xsml_document))
    return XSMLFeatureExtractor(
        mission,
        max_states=max_states,
        inventory_noise_variants=inventory_noise_variants,
        random_seed=random_seed,
    ).extract_training_data()


def export_training_csv(matrices: FeatureMatrices, path: str) -> None:
    """Zapisuje płaskie rekordy migawek i zmienne docelowe w osobnych plikach CSV."""
    _write_csv(matrices.flat_records, Path(path))
    _write_csv(matrices.targets, Path(f"{path}.targets.csv"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Zapisuje plik CSV z cechami boolowskimi."""
    output = frame.copy()
    for column in output.select_dtypes(include="bool").columns:
        output[column] = output[column].map({True: "true", False: "false"})
    output.to_csv(path, index=False)


def export_paired_datasets(
    matrices: FeatureMatrices, output_dir: str | Path, prefix: str
) -> dict[str, Path]:
    """Zapisuje obie macierze cech i etykiety w formacie CSV."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    mission_ids = matrices.flat_records["mission_id"].tolist()
    xsml = matrices.xsml.reset_index()
    flat = matrices.flat.reset_index()
    targets = matrices.targets.reset_index()
    for frame in (xsml, flat, targets):
        frame.insert(1, "mission_id", mission_ids)

    paths = {
        "xsml_features": directory / f"{prefix}_xsml_features.csv",
        "flat_features": directory / f"{prefix}_flat_features.csv",
        "targets": directory / f"{prefix}_targets.csv",
    }
    _write_csv(xsml, paths["xsml_features"])
    _write_csv(flat, paths["flat_features"])
    _write_csv(targets, paths["targets"])
    return paths


def main(argv: list[str] | None = None) -> int:
    """Uruchamia ekstrakcję cech z pliku XSML za pośrednictwem wiersza poleceń."""
    argument_parser = argparse.ArgumentParser(
        description="Generuje macierze cech XSML i flat oraz etykiety podpowiedzi z misji XSML."
    )
    argument_parser.add_argument("mission", type=Path, help="Ścieżka do pliku misji XSML w formacie YAML.")
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Katalog zapisu plików CSV (domyślnie: data/features).",
    )
    argument_parser.add_argument(
        "--prefix",
        help="Prefiks nazw plików wyjściowych (domyślnie: identyfikator misji).",
    )
    argument_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Ziarno generatora pseudolosowego (domyślnie: 42).",
    )
    argument_parser.add_argument(
        "--inventory-noise-variants",
        type=int,
        default=2,
        help="Liczba zaszumionych wariantów ekwipunku na migawkę (domyślnie: 2).",
    )
    argument_parser.add_argument(
        "--max-states",
        type=int,
        default=10_000,
        help="Maksymalna liczba enumerowanych stanów (domyślnie: 10000).",
    )
    args = argument_parser.parse_args(argv)

    try:
        with args.mission.open(encoding="utf-8") as mission_file:
            mission_document = yaml.safe_load(mission_file)
        if not isinstance(mission_document, dict):
            raise ValueError("Dokument misji musi zawierać obiekt YAML na najwyższym poziomie.")
        mission_id = mission_document.get("mission", {}).get("id", args.mission.stem)
        matrices = build_paired_feature_matrices(
            mission_document,
            max_states=args.max_states,
            inventory_noise_variants=args.inventory_noise_variants,
            random_seed=args.random_seed,
        )
        paths = export_paired_datasets(
            matrices,
            args.output_dir,
            args.prefix or mission_id,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        argument_parser.error(str(error))

    print(f"Wygenerowano {len(matrices.xsml)} próbek.")
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())