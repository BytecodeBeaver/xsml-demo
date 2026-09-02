from parser.parser import SimulationScenario
from typing import Dict

class SimulationAction:
    def __init__(self, id, description, command, simulated_output, requires, changes, available=False):
        self.description: str = description
        self.command: str = command
        self.simulated_output: str = simulated_output
        self.requires: dict = requires
        self.inventory_changes: dict = changes["inventory"]
        self.worldstate_changes: dict = changes["worldstate"]
        self.inventory_and_worldstate_changes: dict = changes["inventory_and_worldstate"]
        self.available: bool = available

    @staticmethod
    def _iter_produced_entries(action_data: dict):
        for produced in action_data.get("produces") or []:
            if not isinstance(produced, dict):
                continue
            if "inventory_change" in produced or "simulated_output" in produced:
                yield produced
            else:
                for value in produced.values():
                    if isinstance(value, dict):
                        yield value

    @staticmethod
    def _merge_change_map(data) -> dict:
        if not data:
            return {}
        if isinstance(data, dict):
            return dict(data)
        if isinstance(data, list):
            merged = {}
            for entry in data:
                if isinstance(entry, dict):
                    merged.update(entry)
            return merged
        raise ValueError("inventory_change/environment_change musi byc slownikiem lub lista.")

    @classmethod
    def _collect_changes(cls, action_data: dict) -> dict:
        changes = {
            "inventory": {},
            "worldstate": {},
            "inventory_and_worldstate": {},
        }
        for entry in cls._iter_produced_entries(action_data):
            inventory_change = entry.get("inventory_change") or {}
            change = inventory_change.get("change") or {}
            for scope in changes:
                changes[scope].update(cls._merge_change_map(change.get(scope)))
        return changes


class SimulationGoal:
    def __init__(self, goal_data: dict):
        self.id: str = goal_data["id"]
        self.description: str = goal_data.get("description", "")
        self.requires: dict = goal_data.get("requires") or {}
        self.subgoals: Dict[str, "SimulationGoal"] = {
            subgoal.id: subgoal
            for subgoal in (
                SimulationGoal(subgoal_data)
                for subgoal_data in self._iter_goal_data(goal_data.get("subgoals"))
            )
        }
        self.achieved: bool = False

    @staticmethod
    def _iter_goal_data(goals):
        if not goals:
            return []
        if isinstance(goals, list):
            return goals
        if isinstance(goals, dict):
            return goals.values()
        raise ValueError("Podcele musza byc lista lub slownikiem celow.")

class GameLogic:
    def __init__(self, scenario: SimulationScenario):
        self.scenario = scenario
        self.goals: Dict[str, SimulationGoal] = {
            goal.id: goal
            for goal in (
                SimulationGoal(goal_data)
                for goal_data in self._iter_goal_data(scenario.goals)
            )
        }
        # Przetworzenie wiedzy o akcjach do łatwiejszej formy
        self.actions: Dict[str, SimulationAction] = {
            action_data["id"]: SimulationAction(
                id=action_data["id"],
                description=action_data.get("description"),
                command=action_data.get("command", ""),
                simulated_output=self._get_simulated_output(action_data),
                requires=action_data.get("requires") or {},
                changes=SimulationAction._collect_changes(action_data),
                available=self.are_requirements_fullfilled(action_data.get("requires") or {})
            )
            for action_data in self._iter_action_data(scenario.actions)
        }

        self.update_goal_progress()

        scenario.add_listener(self.on_scenario_state_change)

    @staticmethod
    def _iter_goal_data(goals):
        return SimulationGoal._iter_goal_data(goals)

    @staticmethod
    def _iter_action_data(actions):
        if isinstance(actions, list):
            return actions
        if isinstance(actions, dict):
            return actions.values()
        raise ValueError("Akcje musza byc lista lub slownikiem akcji.")

    @staticmethod
    def _get_simulated_output(action_data: dict) -> str:
        for entry in SimulationAction._iter_produced_entries(action_data):
            if "simulated_output" in entry:
                return str(entry["simulated_output"]).rstrip("\n")
        return ""

    def are_requirements_fullfilled(self, requirements: dict) -> bool:
        for requirement, value in requirements["inventory"].items():
            if self.scenario.get_inventory_variable(requirement) != value:
                return False
        for requirement, value in requirements["worldstate"].items():
            if self.scenario.get_worldstate_variable(requirement) != value:
                return False
        return True

    def execute_action(self, action_id: str) -> SimulationAction:
        action = self.actions.get(action_id)
        if action is None:
            raise ValueError(f"Nieznana akcja: {action_id}")
        if not action.available:
            raise ValueError(f"Akcja '{action_id}' jest obecnie niedostepna.")

        # Ustawienie zmiennych powiadamia listenerow (on_scenario_state_change), co
        # aktualizuje dostepnosc akcji oraz postep celow.
        for variable_name, value in action.inventory_changes.items():
            self.scenario.set_inventory_variable(variable_name, value)
        for variable_name, value in action.worldstate_changes.items():
            self.scenario.set_worldstate_variable(variable_name, value)
        for variable_name, value in action.inventory_and_worldstate_changes.items():
            self.scenario.set_inventory_variable(variable_name, value)
            self.scenario.set_worldstate_variable(variable_name, value)

        return action

    def update_goal_progress(self):
        def update_goal(goal: SimulationGoal) -> bool:
            subgoal_results = [
                update_goal(subgoal) for subgoal in goal.subgoals.values()
            ]
            subgoals_achieved = all(subgoal_results)
            goal.achieved = (
                self.are_requirements_fullfilled(goal.requires) and subgoals_achieved
            )
            return goal.achieved

        for goal in self.goals.values():
            print(f"Zaktualizowano postęp celu {goal.id}. Nowa wartość: {update_goal(goal)}")

    def update_available_actions(self):
        for action_name, action in self.actions.items():
            # Check if action requires are met, if yes mark it as available, otherwise mark it as unavailable    
            action.available = self.are_requirements_fullfilled(action.requires)
            if action.available:
                print(f"Akcja {action_name} jest teraz dostępna.")
            else:
                print(f"Akcja {action_name} jest teraz niedostępna.")

    def on_scenario_state_change(self, scope: str, variable_name: str, value):
        self.update_available_actions()
        self.update_goal_progress()