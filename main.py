import glob
import os
from typing import Dict, Tuple

import yaml
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

from parser.parser import SimulationScenario, XSMLParser
from game.game_logic import GameLogic, SimulationGoal

MISSIONS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'missions')


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


def serialize_goal(goal: SimulationGoal) -> dict:
    return {
        "id": goal.id,
        "description": goal.description,
        "achieved": goal.achieved,
        "subgoals": [serialize_goal(subgoal) for subgoal in goal.subgoals.values()],
    }


def serialize_state(game_logic: GameLogic) -> dict:
    return {
        "goals": [serialize_goal(goal) for goal in game_logic.goals.values()],
        "actions": [
            {"id": action_id, "description": action.description, "available": action.available}
            for action_id, action in game_logic.actions.items()
        ],
    }


def create_app() -> Tuple[Flask, SocketIO]:
    app = Flask(__name__, static_folder='game/html/dist', static_url_path='')
    socketio = SocketIO(app)

    scenario_paths = discover_scenarios()
    # Per-connection game state, keyed by socket session id.
    sessions: Dict[str, GameLogic] = {}

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

    @socketio.on('connect', namespace='/ws/game')
    def handle_connect():
        scenario_id = request.args.get('scenario')
        path = scenario_paths.get(scenario_id)
        if path is None:
            emit('error', {"message": f"Nieznany scenariusz: {scenario_id}"})
            return False

        scenario = load_scenario(path)
        game_logic = GameLogic(scenario)
        sessions[request.sid] = game_logic
        emit('state', serialize_state(game_logic))

    @socketio.on('disconnect', namespace='/ws/game')
    def handle_disconnect():
        sessions.pop(request.sid, None)

    @socketio.on('action', namespace='/ws/game')
    def handle_action(data):
        game_logic = sessions.get(request.sid)
        if game_logic is None:
            emit('error', {"message": "Brak aktywnej sesji gry."})
            return

        action_id = (data or {}).get('action_id')
        action = game_logic.actions.get(action_id)
        if action is None:
            emit('error', {"message": f"Nieznana akcja: {action_id}"})
            return
        if not action.available:
            emit('error', {"message": f"Akcja '{action_id}' jest obecnie niedostepna."})
            return

        for variable_name, value in (action.effects or {}).items():
            game_logic.scenario.set_inventory_variable(variable_name, value)

        emit('state', serialize_state(game_logic))

    return app, socketio


def main():
    app, socketio = create_app()
    socketio.run(app, debug=True, port=5000)


if __name__ == "__main__":
    main()