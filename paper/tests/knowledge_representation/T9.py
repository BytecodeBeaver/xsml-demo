"""
Przypadek testowy: T9
Opis: Cel główny powinien być uznany za nieosiągnięty, jeśli cel główny ma niespełnione warunki, ale wszystkie cele szczegółowe (subgoals) zostały osiągnięte.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Skonfigurowanie w celu głównym 'complete_hardening_of_linux_system' dodatkowego niespełnionego wymagania (np. requires: worldstate: {'formal_report_signed': True}).
3. Wykonanie wszystkich akcji scenariusza ('check_user_password', 'set_strong_root_password', 'check_firewall_status', 'enable_firewall') w celu pełnego zrealizowania wszystkich podcelów.
4. Weryfikacja czy wszystkie podcele uzyskały status achieved == True.
5. Weryfikacja czy warunek własny celu głównego nadal NIE jest spełniony (are_requirements_fullfilled zwraca False).
6. Sprawdzenie statusu celu głównego (achieved == False).

Sukces:
- Wszystkie cele szczegółowe mają status achieved == True.
- Warunki własne celu głównego NIE są spełnione (are_requirements_fullfilled zwraca False).
- Cel główny ma status achieved == False (nie jest uznany za osiągnięty).

Porażka:
- Cel główny zostaje oznaczony jako achieved == True mimo niespełnienia jego bezpośrednich warunków własnych.
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


class TestT9MainGoalUnachievedWhenSubgoalsMetButOwnRequirementsNotMet(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationScenario = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_main_goal_unachieved_when_all_subgoals_met_but_own_requirements_unmet(self):
        main_goal = self.game_logic.goals["complete_hardening_of_linux_system"]

        # Ustawienie niespełnionego wymagania własnego dla celu głównego
        main_goal.requires = {
            "inventory": {},
            "worldstate": {"formal_report_signed": True}
        }

        # Wykonanie wszystkich akcji misji w celu osiągnięcia wszystkich podcelów
        self.game_logic.execute_action("check_user_password")
        self.game_logic.execute_action("set_strong_root_password")
        self.game_logic.execute_action("check_firewall_status")
        self.game_logic.execute_action("enable_firewall")

        # 1. Weryfikacja, że wszystkie podcele zostały osiągnięte
        pwd_goal = main_goal.subgoals["set_strong_root_password"]
        self.assertTrue(pwd_goal.subgoals["identify_weak_root_password"].achieved)
        self.assertTrue(pwd_goal.achieved)

        fw_goal = main_goal.subgoals["enable_firewall"]
        self.assertTrue(fw_goal.subgoals["firewall_disabled_detected"].achieved)
        self.assertTrue(fw_goal.achieved)

        self.assertTrue(all(sub.achieved for sub in main_goal.subgoals.values()))

        # 2. Weryfikacja, że wymaganie własne celu głównego NIE jest spełnione
        self.assertFalse(
            self.game_logic.are_requirements_fullfilled(main_goal.requires),
            "Wymaganie własne celu głównego ('formal_report_signed') nie powinno być spełnione."
        )

        # 3. Weryfikacja, że cel główny NIE jest osiągnięty
        self.assertFalse(
            main_goal.achieved,
            "Cel główny musi być uznany za nieosiągnięty, gdy jego własne warunki nie są spełnione, nawet jeśli wszystkie podcele zostały zrealizowane."
        )


if __name__ == "__main__":
    unittest.main()
