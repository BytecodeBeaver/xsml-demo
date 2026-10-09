from parser.validator import XSMLValidator


class SimulationMission:
    def __init__(self, mission_data: dict):
        self.id: str = mission_data.get("id")
        self.title: str = mission_data.get("title")
        self.description: str = mission_data.get("description")
        self.inventory: dict = mission_data.get("inventory") or {}
        self.worldstate: dict = mission_data.get("worldstate") or {}
        self.goals: dict = mission_data.get("goals")
        self.actions: dict = mission_data.get("actions")
        self.support: list = mission_data.get("support") or []
        self.scenarios: list = mission_data.get("scenarios") or []

        # Observer
        self._listeners = []

    def add_listener(self, listener):
        self._listeners.append(listener)

    def _notify_listeners(self, scope: str, variable_name: str, value):
        for listener in self._listeners:
            listener(scope, variable_name, value)

    def set_inventory_variable(self, variable_name: str, value):
        self.inventory[variable_name] = value
        self._notify_listeners("inventory", variable_name, value)

    def get_inventory_variable(self, variable_name: str):
        return self.inventory.get(variable_name)

    def set_worldstate_variable(self, variable_name: str, value):
        self.worldstate[variable_name] = value
        self._notify_listeners("worldstate", variable_name, value)

    def get_worldstate_variable(self, variable_name: str):
        return self.worldstate.get(variable_name)


class XSMLParser:
    """Buduje obiekt misji z dokumentu sprawdzonego przez XSMLValidator."""

    @staticmethod
    def parse(yaml_data):
        scenario_goals = XSMLValidator.validate(yaml_data)
        mission = yaml_data["mission"]
        scenario = {
            "id": mission["id"],
            "title": mission["title"],
            "description": mission["description"],
            "inventory": mission["inventory"],
            "worldstate": mission["worldstate"],
            "goals": scenario_goals,
            "actions": mission["actions"],
            "support": mission["support"],
            "scenarios": mission["scenarios"],
        }
        return SimulationMission(scenario)

    @staticmethod
    def iter_nested_goals(goals):
        """Zachowuje dotychczasowy helper używany przez kod aplikacji."""
        return XSMLValidator.iter_nested_goals(goals)