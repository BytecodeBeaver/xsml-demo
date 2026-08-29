from parser.parser import SimulationScenario
from typing import Dict

class SimulationAction:
     def __init__(self, id, description, requires, effects, available=False):
        self.description: str = description
        self.requires: dict = requires
        self.effects: dict = effects
        self.available: bool = available


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
                requires=action_data.get("requires") or {},
                effects=action_data.get("effects", {}),
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

    def are_requirements_fullfilled(self, requirements: dict) -> bool:
        for requirement, value in requirements.items():
            if self.scenario.get_simulation_variable(requirement) != value:
                return False
        return True

    def update_goal_progress(self):
        def update_goal(goal: SimulationGoal) -> bool:
            subgoals_achieved = all(
                update_goal(subgoal) for subgoal in goal.subgoals.values()
            )
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