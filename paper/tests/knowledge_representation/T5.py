"""
Przypadek testowy: T5
Opis: Sprawdź czy akcja, której warunki są spełnione jest dostępna dla gracza.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Sprawdzenie akcji o pustych wymaganiach początkowych ('check_user_password', 'check_firewall_status') – powinny być dostępne od razu.
3. Spełnienie warunków dla uprzednio niedostępnej akcji 'set_strong_root_password' poprzez wykonanie 'check_user_password'.
4. Weryfikacja czy po zmianie stanu gry akcja 'set_strong_root_password' staje się dostępna (available == True).
5. Wykonanie akcji 'set_strong_root_password' i potwierdzenie braku błędów.

Sukces:
- Akcje ze spełnionymi wymaganiami mają atrybut available == True i mogą zostać pomyślnie wykonane.

Porażka:
- Akcja ze spełnionymi wymaganiami pozostaje niedostępna (available == False) lub jej wykonanie zgłasza błąd dostępności.
"""

import os
import sys
import unittest
import yaml

# Dodanie katalogu głównego projektu do sys.path, aby umożliwić importy pakietów parser i game
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from parser.parser import XSMLParser, SimulationMission
from game.game_logic import GameLogic

MISSION_PATH = os.path.join(PROJECT_ROOT, "data", "missions", "hardening_linux.yaml")


class TestT5AvailableActionRequirements(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationMission = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_initial_actions_with_empty_requirements_are_available(self):
        # Akcje początkowe bez wymagań powinny być od razu dostępne
        self.assertTrue(
            self.game_logic.actions["check_user_password"].available,
            "Akcja 'check_user_password' o pustych wymaganiach powinna być dostępna."
        )
        self.assertTrue(
            self.game_logic.actions["check_firewall_status"].available,
            "Akcja 'check_firewall_status' o pustych wymaganiach powinna być dostępna."
        )

    def test_action_becomes_available_once_requirements_are_met(self):
        # Przed spełnieniem warunku
        self.assertFalse(self.game_logic.actions["set_strong_root_password"].available)

        # Wykonanie akcji 'check_user_password' spełniającej wymaganie
        self.game_logic.execute_action("check_user_password")

        # Po spełnieniu warunku
        self.assertTrue(
            self.game_logic.actions["set_strong_root_password"].available,
            "Akcja 'set_strong_root_password' powinna stać się dostępna po wykryciu słabego hasła."
        )

        # Sprawdzenie czy akcja może zostać poprawnie wykonana
        action = self.game_logic.execute_action("set_strong_root_password")
        self.assertIsNotNone(action)


if __name__ == "__main__":
    unittest.main()
