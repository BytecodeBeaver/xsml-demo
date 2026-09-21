import glob
import os
import re
from typing import Dict, Tuple

import yaml
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

from parser.parser import SimulationScenario, XSMLParser
from game.game_logic import GameLogic, SimulationGoal

MISSIONS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'missions')
NICKNAME_PATTERN = re.compile(r'^[A-Za-z0-9 _-]{1,24}$')


def discover_scenarios() -> Dict[str, str]:
    """Map scenario id -> yaml file path, skipping files that fail to parse."""
    scenarios: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(MISSIONS_DIR, '*.yaml'))):
        try:
            scenario = load_scenario(path)
        except Exception:
            continue
        scenarios[scenario.id] = path
    return scenarios


def load_scenario(path: str) -> SimulationScenario:
    with open(path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    return XSMLParser.parse(yaml_data)


def serialize_goal(goal: SimulationGoal, is_top_level: bool = False) -> dict:
    return {
        "id": goal.id,
        # The top-level goal's content is always known; deeper goals stay hidden until achieved.
        "description": goal.description if (is_top_level or goal.achieved) else None,
        "achieved": goal.achieved,
        "subgoals": [serialize_goal(subgoal) for subgoal in goal.subgoals.values()],
    }


def serialize_state(game_logic: GameLogic) -> dict:
    return {
        "goals": [serialize_goal(goal, is_top_level=True) for goal in game_logic.goals.values()],
        "actions": [
            {"id": action_id, "description": action.description, "available": action.available}
            for action_id, action in game_logic.actions.items()
        ],
    }


def serialize_hints(game_logic: GameLogic) -> dict:
    return {
        "hints": [
            {"id": hint.id, "hint": hint.hint}
            for hint in game_logic.get_available_hints()
        ],
    }


def build_walkthroughs(scenario: SimulationScenario) -> list:
    """Turn each mission.scenarios graph into a linear, ordered list of steps
    (with alternative actions grouped per step) leading from start to goal."""
    actions_by_id = {action["id"]: action for action in scenario.actions}
    goal_descriptions = {
        goal["id"]: goal["description"]
        for goal in XSMLParser.iter_nested_goals(scenario.goals)
    }
    walkthroughs = []
    for walkthrough in scenario.scenarios:
        nodes = walkthrough["nodes"]
        transitions = walkthrough["transitions"]

        transitions_from: Dict[str, list] = {}
        target_nodes = set()
        for transition in transitions:
            transitions_from.setdefault(transition["from"], []).append(transition)
            target_nodes.add(transition["to"])

        start_candidates = [node_id for node_id in nodes if node_id not in target_nodes]
        current = start_candidates[0] if start_candidates else next(iter(nodes), None)

        steps = []
        visited = set()
        while current is not None and current in transitions_from and current not in visited:
            visited.add(current)
            group = transitions_from[current]
            next_node = group[0]["to"]
            steps.append({
                "from_description": nodes[current]["description"],
                "to_description": nodes[next_node]["description"],
                "actions": [
                    {
                        "id": transition["action_trigger"],
                        "description": actions_by_id.get(transition["action_trigger"], {}).get("description", transition["action_trigger"]),
                    }
                    for transition in group
                ],
            })
            current = next_node

        walkthroughs.append({
            "id": walkthrough["id"],
            "goal": walkthrough["goal"],
            "goal_description": goal_descriptions.get(walkthrough["goal"], walkthrough["goal"]),
            "steps": steps,
        })
    return walkthroughs


def create_app() -> Tuple[Flask, SocketIO]:
    app = Flask(__name__, static_folder='game/html/dist', static_url_path='')
    socketio = SocketIO(app)

    scenario_paths = discover_scenarios()
    # Game state per (nickname, scenario_id), so progress survives reconnects and is
    # separated between players sharing the same scenario.
    player_sessions: Dict[Tuple[str, str], GameLogic] = {}
    # Maps a live socket connection to the player session it is currently attached to.
    sid_to_key: Dict[str, Tuple[str, str]] = {}

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/api/scenarios')
    def list_scenarios():
        scenarios = []
        for path in scenario_paths.values():
            scenario = load_scenario(path)
            scenarios.append({
                "id": scenario.id,
                "title": scenario.title,
                "description": scenario.description,
            })
        return jsonify(scenarios)

    @app.route('/api/scenarios/<scenario_id>/walkthrough')
    def get_scenario_walkthrough(scenario_id):
        path = scenario_paths.get(scenario_id)
        if path is None:
            return jsonify({"message": f"Nieznany scenariusz: {scenario_id}"}), 404
        scenario = load_scenario(path)
        return jsonify(build_walkthroughs(scenario))

    @socketio.on('connect', namespace='/ws/game')
    def handle_connect():
        scenario_id = request.args.get('scenario')
        path = scenario_paths.get(scenario_id)
        if path is None:
            emit('error', {"message": f"Nieznany scenariusz: {scenario_id}"})
            return False

        nickname = (request.args.get('nickname') or '').strip()
        if not NICKNAME_PATTERN.match(nickname):
            emit('error', {"message": "Nieprawidlowy nick (1-24 znakow: litery, cyfry, spacja, - lub _)."})
            return False

        key = (nickname, scenario_id)
        game_logic = player_sessions.get(key)
        if game_logic is None:
            game_logic = GameLogic(load_scenario(path))
            player_sessions[key] = game_logic

        sid_to_key[request.sid] = key
        emit('state', serialize_state(game_logic))
        emit('hints', serialize_hints(game_logic))

    @socketio.on('disconnect', namespace='/ws/game')
    def handle_disconnect():
        # Player progress is kept in player_sessions so the same nickname can resume later.
        sid_to_key.pop(request.sid, None)

    @socketio.on('action', namespace='/ws/game')
    def handle_action(data):
        key = sid_to_key.get(request.sid)
        game_logic = player_sessions.get(key) if key else None
        if game_logic is None:
            emit('error', {"message": "Brak aktywnej sesji gry."})
            return

        action_id = (data or {}).get('action_id')
        try:
            action = game_logic.execute_action(action_id)
        except ValueError as error:
            emit('error', {"message": str(error)})
            return

        emit('terminal_output', {
            "user_input": action.user_input,
            "simulated_output": action.simulated_output,
        })
        emit('state', serialize_state(game_logic))
        emit('hints', serialize_hints(game_logic))

    @socketio.on('hints_request', namespace='/ws/game')
    def handle_hints_request():
        key = sid_to_key.get(request.sid)
        game_logic = player_sessions.get(key) if key else None
        if game_logic is not None:
            emit('hints', serialize_hints(game_logic))

    return app, socketio


def main():
    app, socketio = create_app()
    socketio.run(app, debug=True, port=5000)


if __name__ == "__main__":
    main()