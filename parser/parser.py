class SimulationScenario:
    def __init__(self, scenario_data: dict):
        self.id: str = scenario_data.get("id")
        self.title: str = scenario_data.get("title")
        self.description: str = scenario_data.get("description")
        self.inventory: dict = scenario_data.get("inventory")
        self.environment: dict = scenario_data.get("environment")
        self.goals: dict = scenario_data.get("goals")
        self.actions: dict = scenario_data.get("actions")

class XSMLParser:
    @staticmethod
    def construct_goal_tree(goals):
        # Funkcja rekurencyjna do konstruowania drzewa celów
        def build_tree(goal):
            if isinstance(goal, dict):
                return {k: build_tree(v) for k, v in goal.items()}
            elif isinstance(goal, list):
                return [build_tree(g) for g in goal]
            else:
                return goal

        return build_tree(goals)

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
            raise ValueError("Brak tytułu scenariusza w sekcji 'scenario'.")

        scenario_description = scenario_root.get("description")
        if not scenario_description:
            raise ValueError("Brak opisu scenariusza w sekcji 'scenario'.")

        if "inventory" not in scenario_root:
            raise ValueError("Brak sekcji 'inventory' w scenariuszu.")
        scenario_inventory = scenario_root["inventory"]

        if "environment" not in scenario_root:
            raise ValueError("Brak sekcji 'environment' w scenariuszu.")
        scenario_environment = scenario_root["environment"]

        scenario_goals = scenario_root.get("goals", {})
        if not scenario_goals:
            raise ValueError("Brak sekcji 'goals' w scenariuszu.")

        scenario_actions = scenario_root.get("actions", {})
        if not scenario_actions:
            raise ValueError("Brak sekcji 'actions' w scenariuszu.")
    
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