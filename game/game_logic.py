from parser.parser import SimulationScenario
from typing import Dict

class SimulationAction:
     def __init__(self, id, description, requires, effects, available=False):
        self.description: str = description
        self.requires: dict = requires
        self.effects: dict = effects
        self.available: bool = available

class GameLogic:
    def __init__(self, scenario: SimulationScenario):
        self.scenario = scenario
        self.goals = scenario.goals
        self.actions: Dict[str, SimulationAction] = {
            action_name: SimulationAction(
                id=action_name,
                description=action_data.get("description"),
                requires=action_data.get("requires", {}),
                effects=action_data.get("effects", {}),
                available=True if self.check_action_requirements(action_data.get("requires", {})) else False
            )
            for action_name, action_data in scenario.actions.items()
        }

        scenario.add_listener(self.on_scenario_state_change)

    def check_action_requirements(self, requirements: dict) -> bool:
        for requirement, value in requirements.items():
            if self.scenario.get_simulation_variable(requirement) != value:
                return False
        return True

    def update_goal_progress(self):
        # TODO: Implement goal progress update logic

        pass

    def update_available_actions(self):
        for action_name, action in self.actions.items():
            # Check if action requires are met, if yes mark it as available, otherwise mark it as unavailable    
                # Action is available
                pass
        

    def on_scenario_state_change(self, scope: str, variable_name: str, value):
        self.update_available_actions()
        self.update_goal_progress()