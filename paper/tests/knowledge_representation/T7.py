"""
Przypadek testowy: T7
Opis: Sprawdź czy cel główny jest uznany za nieosiągnięty, jeśli są spełnione warunki celu głównego, ale cele szczegółowe (subgoals) nie są osiągnięte.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Weryfikacja, że warunki własne celu głównego 'complete_hardening_of_linux_system' są spełnione (requires: inventory: {}, worldstate: {}).
3. Sprawdzenie, że podcele celu głównego nie zostały jeszcze zrealizowane (lub zrealizowano tylko część z nich).
4. Sprawdzenie wartości atrybutu achieved dla celu głównego.

Sukces:
- Warunki własne celu głównego zwracają True (are_requirements_fullfilled).
- Co najmniej jeden cel szczegółowy ma status achieved == False.
- Cel główny ma status achieved == False (nie jest uznany za osiągnięty).

Porażka:
- Cel główny zostaje oznaczony jako achieved == True mimo braku realizacji jego podcelów.
"""

import os
import sys
import unittest
import yaml

# Dodanie katalogu głównego projektu do sys.path, aby umożliwić importy pakietów parser i game
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from parser.parser import XSMLParser, SimulationScenario
from game.game_logic import GameLogic

MISSION_PATH = os.path.join(PROJECT_ROOT, "data", "missions", "hardening_linux.yaml")


class TestT7MainGoalUnachievedWhenSubgoalsIncomplete(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationScenario = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_main_goal_unachieved_at_start_even_with_fulfilled_own_requirements(self):
        main_goal = self.game_logic.goals["complete_hardening_of_linux_system"]

        # Warunki własne celu głównego są spełnione (puste słowniki wymagań)
        self.assertTrue(
            self.game_logic.are_requirements_fullfilled(main_goal.requires),
            "Własne wymagania celu głównego powinny być spełnione."
        )

        # Żaden podcel nie został osiągnięty
        for subgoal in main_goal.subgoals.values():
            self.assertFalse(subgoal.achieved, f"Podcel '{subgoal.id}' nie powinien być osiągnięty na starcie.")

        # Cel główny musi pozostać nieosiągnięty
        self.assertFalse(
            main_goal.achieved,
            "Cel główny nie może być uznany za osiągnięty, dopóki jego podcele nie są zrealizowane."
        )

    def test_main_goal_unachieved_when_only_partial_subgoals_completed(self):
        main_goal = self.game_logic.goals["complete_hardening_of_linux_system"]

        # Realizacja tylko części ścieżki (tylko gałąź zmiany hasła)
        self.game_logic.execute_action("check_user_password")
        self.game_logic.execute_action("set_strong_root_password")

        # Gałąź hasła jest osiągnięta
        self.assertTrue(main_goal.subgoals["set_strong_root_password"].achieved)

        # Gałąź zapory sieciowej NIE jest osiągnięta
        self.assertFalse(main_goal.subgoals["enable_firewall"].achieved)

        # Cel główny nadal powinien być nieosiągnięty
        self.assertFalse(
            main_goal.achieved,
            "Cel główny nie powinien być osiągnięty, gdy część podcelów pozostaje niezrealizowana."
        )


if __name__ == "__main__":
    unittest.main()
