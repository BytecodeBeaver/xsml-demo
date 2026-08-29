class SimulationScenario:
    def __init__(self, scenario_data: dict):
        self.id: str = scenario_data.get("id")
        self.title: str = scenario_data.get("title")
        self.description: str = scenario_data.get("description")
        self.inventory: dict = scenario_data.get("inventory")
        self.environment: dict = scenario_data.get("environment")
        self.goals: dict = scenario_data.get("goals")
        self.actions: dict = scenario_data.get("actions")

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

    def set_environment_variable(self, variable_name: str, value):
        self.environment[variable_name] = value
        self._notify_listeners("environment", variable_name, value)

    def get_environment_variable(self, variable_name: str):
        return self.environment.get(variable_name)

    def get_simulation_variable(self, variable_name: str):
        # Albo zwróć wartość z inventory, albo z environment, jeśli nie ma w inventory
        inventory_value = self.inventory.get(variable_name)
        if inventory_value is not None:
            return inventory_value
        else:
            return self.environment.get(variable_name)

class XSMLParser:
    @staticmethod
    def construct_goal_tree(goals):
        def build_tree(goal, path="goals"):
            if isinstance(goal, list):
                return [build_tree(subgoal, f"{path}[{index}]") for index, subgoal in enumerate(goal)]

            if not isinstance(goal, dict):
                raise ValueError(f"Cel '{path}' jest blednie zdefiniowany.")

            missing_sections = [
                section for section in ("id", "description", "requires") if section not in goal
            ]
            if missing_sections:
                goal_name = goal.get("id", path)
                sections = ", ".join(f"'{section}'" for section in missing_sections)
                raise ValueError(
                    f"Cel '{goal_name}' jest blednie zdefiniowany: brak sekcji {sections}."
                )

            goal_tree = dict(goal)
            subgoals = goal.get("subgoals")
            if subgoals:
                goal_tree["subgoals"] = build_tree(
                    subgoals, f"{path}.{goal['id']}.subgoals"
                )
            else:
                goal_tree["subgoals"] = {}
            return goal_tree

        if isinstance(goals, list):
            return build_tree(goals)

        if isinstance(goals, dict):
            return {
                goal_id: build_tree(goal, f"goals.{goal_id}")
                for goal_id, goal in goals.items()
            }

        raise ValueError("Sekcja 'goals' musi byc lista celow lub slownikiem celow.")

    @staticmethod
    def parse(yaml_data):
        # Sprawdzenie metadanych w celu ustalenia wersji i typu scenariusza
        meta = yaml_data.get("meta", {})
        if meta.get("dialect") != "xsml":
            raise ValueError("Nieprawidłowy dialekt scenariusza. Oczekiwano 'xsml'.")
        if meta.get("version") != "0.1": # TODO: W przyszłości obsłużyć różne wersje scenariuszy
            raise ValueError("Nieprawidłowa wersja scenariusza. Parser wspiera tylko wersję '0.1'.")

        scenario_root = yaml_data.get("scenario", {})
        if not scenario_root:
            raise ValueError("Brak sekcji 'scenario' w scenariuszu.")

        scenario_id = scenario_root.get("id")
        if not scenario_id:
            raise ValueError("Brak identyfikatora scenariusza w sekcji 'scenario'.")

        scenario_title = scenario_root.get("title")
        if not scenario_title:
            raise ValueError("Niezdefiniowano tytułu scenariusza w sekcji 'scenario'.")

        scenario_description = scenario_root.get("description")
        if not scenario_description:
            raise ValueError("Niezdefiniowano opisu scenariusza w sekcji 'scenario'.")

        if "inventory" not in scenario_root:
            raise ValueError("Brak sekcji 'inventory' w scenariuszu.")
        scenario_inventory = scenario_root["inventory"]

        if "environment" not in scenario_root:
            raise ValueError("Brak sekcji 'environment' w scenariuszu.")
        scenario_environment = scenario_root["environment"]

        scenario_goals = scenario_root.get("goals", {})
        if not scenario_goals:
            raise ValueError("Brak sekcji lub pusta sekcja 'goals' w scenariuszu.")

        scenario_actions = scenario_root.get("actions", {})
        if not scenario_actions:
            raise ValueError("Brak sekcji lub pusta sekcja 'actions' w scenariuszu.")
    
        scenario = {
            "id": scenario_id,
            "title": str(scenario_title),
            "description": str(scenario_description),
            "inventory": scenario_inventory,
            "environment": scenario_environment,
            "goals": XSMLParser.construct_goal_tree(scenario_goals),
            "actions": scenario_actions
        }

        return SimulationScenario(scenario)