"""
Przypadek testowy: T6
Opis: Sprawdź czy cel główny jest uznany za osiągnięty, jeśli cele szczegółowe (subgoals) zostały osiągnięte i warunek celu głównego także został osiągnięty.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Sprawdzenie stanu początkowego celu głównego 'complete_hardening_of_linux_system' (achieved == False).
3. Wykonanie pełnej sekwencji akcji realizującej wszystkie podcele:
   - 'check_user_password' -> realizuje podcel 'identify_weak_root_password'
   - 'set_strong_root_password' -> realizuje podcel 'set_strong_root_password'
   - 'check_firewall_status' -> realizuje podcel 'firewall_disabled_detected'
   - 'enable_firewall' -> realizuje podcel 'enable_firewall'
4. Weryfikacja osiągnięcia celów szczegółowych.
5. Weryfikacja warunków własnych celu głównego (requires: inventory: {}, worldstate: {}).
6. Sprawdzenie czy cel główny posiada status achieved == True.

Sukces:
- Wszystkie podcele oraz cel główny mają atrybut achieved == True.

Porażka:
- Cel główny ma status achieved == False mimo osiągnięcia wszystkich podcelów i spełnienia własnych warunków.
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


class TestT6MainGoalAchievedWhenRequirementsAndSubgoalsMet(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationScenario = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_main_goal_achieved_after_all_subgoals_and_requirements_fulfilled(self):
        main_goal = self.game_logic.goals["complete_hardening_of_linux_system"]
        self.assertFalse(main_goal.achieved, "Cel główny początkowo nie powinien być osiągnięty.")

        # Wykonanie wszystkich akcji misji
        self.game_logic.execute_action("check_user_password")
        self.game_logic.execute_action("set_strong_root_password")
        self.game_logic.execute_action("check_firewall_status")
        self.game_logic.execute_action("enable_firewall")

        # Weryfikacja podcelu 'set_strong_root_password' i jego zagnieżdżonego podcelu
        pwd_goal = main_goal.subgoals["set_strong_root_password"]
        self.assertTrue(pwd_goal.subgoals["identify_weak_root_password"].achieved)
        self.assertTrue(pwd_goal.achieved)

        # Weryfikacja podcelu 'enable_firewall' i jego zagnieżdżonego podcelu
        fw_goal = main_goal.subgoals["enable_firewall"]
        self.assertTrue(fw_goal.subgoals["firewall_disabled_detected"].achieved)
        self.assertTrue(fw_goal.achieved)

        # Weryfikacja warunków własnych celu głównego
        self.assertTrue(self.game_logic.are_requirements_fullfilled(main_goal.requires))

        # Weryfikacja osiągnięcia celu głównego
        self.assertTrue(
            main_goal.achieved,
            "Cel główny powinien być oznaczony jako osiągnięty (achieved == True)."
        )


if __name__ == "__main__":
    unittest.main()
