import re


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
        self.scenarios: list = scenario_data.get("scenarios") or []

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
        if requires is None:
            requires = {}
        if not isinstance(requires, dict):
            raise ValueError(f"Wymagania '{owner}' musza byc slownikiem.")

        normalized = {}
        for scope in ("inventory", "worldstate"):
            scope_values = requires.get(scope, {})
            if not isinstance(scope_values, dict):
                raise ValueError(f"Sekcja '{scope}' w wymaganiach '{owner}' musi byc slownikiem.")
            normalized[scope] = scope_values
        return normalized

    @staticmethod
    def construct_goal_tree(goals):
        def build_tree(goal, path="goals"):
            if isinstance(goal, list):
                return [build_tree(subgoal, f"{path}[{index}]") for index, subgoal in enumerate(goal)]

            if not isinstance(goal, dict):
                raise ValueError(f"Cel '{path}' jest blednie zdefiniowany.")

            missing_sections = [section for section in ("id", "description", "requires") if section not in goal]
            if missing_sections:
                goal_name = goal.get("id", path)
                sections = ", ".join(f"'{section}'" for section in missing_sections)
                raise ValueError(
                    f"Cel '{goal_name}' jest blednie zdefiniowany: brak sekcji {sections}."
                )

            if not isinstance(goal["id"], str) or not goal["id"]:
                raise ValueError(f"Cel '{path}' musi miec tekstowy identyfikator.")
            if not isinstance(goal["description"], str):
                raise ValueError(f"Opis celu '{goal['id']}' musi byc tekstem.")
            goal["requires"] = XSMLParser.validate_requires(goal["requires"], goal["id"])

            goal_tree = dict(goal)
            subgoals = goal.get("subgoals")
            if subgoals is None:
                goal_tree["subgoals"] = []
            else:
                if not isinstance(subgoals, list):
                    raise ValueError(f"Podcele celu '{goal['id']}' musza byc lista.")
                goal_tree["subgoals"] = build_tree(subgoals, f"{path}.{goal['id']}.subgoals")
            return goal_tree

        if not isinstance(goals, list) or not goals:
            raise ValueError("Sekcja 'goals' musi byc niepusta lista celow.")
        result = build_tree(goals)
        goal_ids = [goal["id"] for goal in XSMLParser.iter_nested_goals(result)]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("Identyfikatory celow musza byc unikalne.")
        return result

    @staticmethod
    def iter_nested_goals(goals):
        for goal in goals:
            yield goal
            yield from XSMLParser.iter_nested_goals(goal.get("subgoals", []))

    @staticmethod
    def validate_actions(actions):
        if not isinstance(actions, list) or not actions:
            raise ValueError("Sekcja 'actions' musi byc niepusta lista akcji.")
        action_data = actions
        action_ids = []
        for action in action_data:
            if not isinstance(action, dict):
                raise ValueError("Akcja musi byc slownikiem.")
            XSMLParser.validate_required_fields(action, ("id", "description", "user_input", "requires", "produces"), "Akcja")
            action_id = action["id"]
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("Identyfikator akcji musi byc tekstem.")
            if not isinstance(action["description"], str):
                raise ValueError(f"Opis akcji '{action_id}' musi byc tekstem.")
            action["requires"] = XSMLParser.validate_requires(action["requires"], action_id)
            produces = action["produces"]
            if not isinstance(produces, list) or not produces:
                raise ValueError(f"Rezultaty akcji '{action_id}' musza byc niepusta lista.")
            probabilities = 0.0
            result_ids = []
            for produced in produces:
                XSMLParser.validate_required_fields(
                    produced,
                    ("id", "probability", "change", "simulated_output"),
                    f"Rezultat akcji '{action_id}'",
                )
                if not isinstance(produced["id"], str) or not produced["id"]:
                    raise ValueError(f"Identyfikator rezultatu akcji '{action_id}' musi byc tekstem.")
                if not isinstance(produced["probability"], float) or not 0.0 <= produced["probability"] <= 1.0:
                    raise ValueError(f"Prawdopodobienstwo rezultatu '{produced['id']}' musi byc liczba zmiennoprzecinkowa od 0 do 1.")
                change = produced["change"]
                if not isinstance(change, dict):
                    raise ValueError(f"Zmiana rezultatu '{produced['id']}' musi byc mapa.")
                for scope in ("inventory", "worldstate", "inventory_and_worldstate"):
                    if not isinstance(change.get(scope), dict):
                        raise ValueError(f"Zmiana rezultatu '{produced['id']}' musi zawierac mape '{scope}'.")
                probabilities += produced["probability"]
                result_ids.append(produced["id"])
            if len(result_ids) != len(set(result_ids)):
                raise ValueError(f"Rezultaty akcji '{action_id}' musza miec unikalne identyfikatory.")
            if abs(probabilities - 1.0) > 1e-9:
                raise ValueError(f"Prawdopodobienstwa rezultatow akcji '{action_id}' musza sumowac sie do 1.0.")
            action_ids.append(action_id)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Identyfikatory akcji musza byc unikalne.")

    @staticmethod
    def validate_required_fields(value, fields, owner):
        if not isinstance(value, dict):
            raise ValueError(f"{owner} musi byc mapa.")
        missing = [field for field in fields if field not in value]
        if missing:
            raise ValueError(f"{owner} nie zawiera wymaganych pol: {', '.join(missing)}.")

    @staticmethod
    def validate_support(support):
        if support is None:
            return
        if not isinstance(support, list):
            raise ValueError("Sekcja 'support' musi byc lista.")

        problem_ids = []
        for problem in support:
            XSMLParser.validate_required_fields(problem, ("id", "hint", "requires"), "Problem gracza")
            if not isinstance(problem["id"], str) or not problem["id"] or not isinstance(problem["hint"], str):
                raise ValueError("Identyfikator i podpowiedz problemu gracza musza byc tekstami.")
            problem["requires"] = XSMLParser.validate_requires(problem["requires"], problem["id"])

            idle_time = (problem["requires"].get("inventory") or {}).get("player_idle_time")
            if idle_time is not None and (not isinstance(idle_time, (int, float)) or isinstance(idle_time, bool) or idle_time < 0):
                raise ValueError("player_idle_time musi byc nieujemna liczba sekund.")
            problem_ids.append(problem["id"])
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError("Identyfikatory problemow gracza musza byc unikalne.")

    @staticmethod
    def validate_scenarios(scenarios, goal_ids, action_ids):
        if not isinstance(scenarios, list):
            raise ValueError("Sekcja 'scenarios' musi byc lista.")
        scenario_ids = []
        for scenario in scenarios:
            XSMLParser.validate_required_fields(scenario, ("id", "goal", "nodes", "transitions"), "Scenariusz")
            if not all(isinstance(scenario[field], str) and scenario[field] for field in ("id", "goal")):
                raise ValueError("Identyfikator scenariusza i jego cel musza byc tekstami.")
            if scenario["goal"] not in goal_ids:
                raise ValueError(f"Scenariusz '{scenario['id']}' wskazuje nieistniejacy cel.")
            nodes = scenario["nodes"]
            transitions = scenario["transitions"]
            if not isinstance(nodes, dict) or not isinstance(transitions, list):
                raise ValueError(f"Scenariusz '{scenario['id']}' ma niepoprawne wezly lub przejscia.")
            node_ids = []
            for node_key, node in nodes.items():
                XSMLParser.validate_required_fields(node, ("id", "description"), "Wezel scenariusza")
                if not isinstance(node["id"], str) or not node["id"] or not isinstance(node["description"], str):
                    raise ValueError(f"Wezel scenariusza '{node_key}' ma niepoprawne pola.")
                node_ids.append(node["id"])
            if len(node_ids) != len(set(node_ids)):
                raise ValueError(f"Wezly scenariusza '{scenario['id']}' musza miec unikalne identyfikatory.")
            for transition in transitions:
                XSMLParser.validate_required_fields(transition, ("from", "to", "action_trigger"), "Przejscie scenariusza")
                if transition["from"] not in node_ids or transition["to"] not in node_ids:
                    raise ValueError(f"Przejscie scenariusza '{scenario['id']}' wskazuje nieistniejacy wezel.")
                if transition["action_trigger"] not in action_ids:
                    raise ValueError(f"Przejscie scenariusza '{scenario['id']}' wskazuje nieistniejaca akcje.")
            scenario_ids.append(scenario["id"])
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Identyfikatory scenariuszy musza byc unikalne.")

    @staticmethod
    def parse(yaml_data):
        if not isinstance(yaml_data, dict):
            raise ValueError("Dokument XSML musi byc mapa.")
        XSMLParser.validate_required_fields(yaml_data, ("meta", "mission"), "Dokument XSML")
        meta = yaml_data["meta"]
        XSMLParser.validate_required_fields(meta, ("type", "version"), "Sekcja 'meta'")
        if meta["type"] != "xsml" or not isinstance(meta["type"], str):
            raise ValueError("Pole 'meta.type' musi byc tekstem 'xsml'.")
        if not isinstance(meta["version"], str) or not re.fullmatch(r"\d+\.\d+\.\d+", meta["version"]):
            raise ValueError("Pole 'meta.version' musi miec format <liczba>.<liczba>.<liczba>.")

        mission = yaml_data["mission"]
        required_mission_fields = ("id", "title", "description", "worldstate", "inventory", "goals", "actions", "support", "scenarios")
        XSMLParser.validate_required_fields(mission, required_mission_fields, "Sekcja 'mission'")
        for field in ("id", "title", "description"):
            if not isinstance(mission[field], str) or not mission[field]:
                raise ValueError(f"Pole 'mission.{field}' musi byc niepustym tekstem.")
        for field in ("worldstate", "inventory"):
            if not isinstance(mission[field], dict):
                raise ValueError(f"Pole 'mission.{field}' musi byc mapa.")

        scenario_goals = XSMLParser.construct_goal_tree(mission["goals"])
        XSMLParser.validate_actions(mission["actions"])
        XSMLParser.validate_support(mission["support"])
        goal_ids = {goal["id"] for goal in XSMLParser.iter_nested_goals(scenario_goals)}
        action_ids = {action["id"] for action in mission["actions"]}
        XSMLParser.validate_scenarios(mission["scenarios"], goal_ids, action_ids)

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

        return SimulationScenario(scenario)