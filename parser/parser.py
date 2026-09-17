class SimulationScenario:
    def __init__(self, scenario_data: dict):
        self.id: str = scenario_data.get("id")
        self.title: str = scenario_data.get("title")
        self.description: str = scenario_data.get("description")
        self.inventory: dict = scenario_data.get("inventory") or {}
        self.worldstate: dict = scenario_data.get("worldstate") or {}
        self.goals: dict = scenario_data.get("goals")
        self.actions: dict = scenario_data.get("actions")
        self.support: list = scenario_data.get("support") or []

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
    @staticmethod
    def validate_requires(requires, owner: str):
        if not isinstance(requires, dict):
            raise ValueError(f"Wymagania '{owner}' musza byc slownikiem.")

        expected_scopes = {"inventory", "worldstate"}
        if set(requires) != expected_scopes:
            raise ValueError(
                f"Wymagania '{owner}' musza zawierac sekcje 'inventory' i 'worldstate'."
            )

        for scope in expected_scopes:
            if not isinstance(requires[scope], dict):
                raise ValueError(f"Sekcja '{scope}' w wymaganiach '{owner}' musi byc slownikiem.")

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

            XSMLParser.validate_requires(goal["requires"], goal.get("id", path))

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
    def validate_actions(actions):
        action_data = actions if isinstance(actions, list) else actions.values()
        for action in action_data:
            if not isinstance(action, dict):
                raise ValueError("Akcja musi byc slownikiem.")
            XSMLParser.validate_requires(action.get("requires"), action.get("id", "actions"))
            for produced in action.get("produces") or []:
                inventory_change = produced.get("inventory_change")
                if inventory_change is None:
                    continue
                change = inventory_change.get("change") if isinstance(inventory_change, dict) else None
                expected_scopes = {"inventory", "worldstate", "inventory_and_worldstate"}
                if not isinstance(change, dict) or set(change) != expected_scopes:
                    raise ValueError(
                        "inventory_change musi zawierac sekcje 'change' z zakresami "
                        "'inventory', 'worldstate' i 'inventory_and_worldstate'."
                    )
                for scope in expected_scopes:
                    if not isinstance(change[scope], dict):
                        raise ValueError(f"Sekcja '{scope}' w zmianie musi byc slownikiem.")

    @staticmethod
    def validate_support(support):
        if support is None:
            return
        if not isinstance(support, list):
            raise ValueError("Sekcja 'support' musi byc lista.")

        for problem in support:
            if not isinstance(problem, dict) or not problem.get("id") or "hint" not in problem:
                raise ValueError("Kazda podpowiedz musi zawierac 'id' i 'hint'.")
            XSMLParser.validate_requires(problem.get("requires"), problem["id"])

            idle_time = (problem["requires"].get("inventory") or {}).get("player_idle_time")
            if idle_time is not None and (not isinstance(idle_time, (int, float)) or isinstance(idle_time, bool) or idle_time < 0):
                raise ValueError("player_idle_time musi byc nieujemna liczba sekund.")

    @staticmethod
    def parse(yaml_data):
        # Sprawdzenie metadanych w celu ustalenia wersji i typu scenariusza
        meta = yaml_data.get("meta", {})
        if meta.get("dialect") != "xsml":
            raise ValueError("Nieprawidłowy dialekt scenariusza. Oczekiwano 'xsml'.")
        if meta.get("version") != "0.1": # TODO: W przyszłości obsłużyć różne wersje scenariuszy
            raise ValueError("Nieprawidłowa wersja scenariusza. Parser wspiera tylko wersję '0.1'.")

        scenario_root = yaml_data.get("mission", {})
        if not scenario_root:
            raise ValueError("Brak sekcji 'mission' w scenariuszu.")

        scenario_id = scenario_root.get("id")
        if not scenario_id:
            raise ValueError("Brak identyfikatora scenariusza w sekcji 'mission'.")

        scenario_title = scenario_root.get("title")
        if not scenario_title:
            raise ValueError("Niezdefiniowano tytułu scenariusza w sekcji 'mission'.")

        scenario_description = scenario_root.get("description")
        if not scenario_description:
            raise ValueError("Niezdefiniowano opisu scenariusza w sekcji 'mission'.")

        if "inventory" not in scenario_root:
            raise ValueError("Brak sekcji 'inventory' w scenariuszu.")
        scenario_inventory = scenario_root["inventory"]

        if "worldstate" not in scenario_root:
            raise ValueError("Brak sekcji 'worldstate' w scenariuszu.")
        scenario_worldstate = scenario_root["worldstate"]

        scenario_goals = scenario_root.get("goals", {})
        if not scenario_goals:
            raise ValueError("Brak sekcji lub pusta sekcja 'goals' w scenariuszu.")

        scenario_actions = scenario_root.get("actions", {})
        if not scenario_actions:
            raise ValueError("Brak sekcji lub pusta sekcja 'actions' w scenariuszu.")
        XSMLParser.validate_actions(scenario_actions)

        scenario_support = scenario_root.get("support") or []
        XSMLParser.validate_support(scenario_support)
    
        scenario = {
            "id": scenario_id,
            "title": str(scenario_title),
            "description": str(scenario_description),
            "inventory": scenario_inventory,
            "worldstate": scenario_worldstate,
            "goals": XSMLParser.construct_goal_tree(scenario_goals),
            "actions": scenario_actions,
            "support": scenario_support,
        }

        return SimulationScenario(scenario)