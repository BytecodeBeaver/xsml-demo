"""
Przypadek testowy: T4
Opis: Sprawdź czy akcja, której warunki nie są spełnione nie jest dostępna dla gracza.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Weryfikacja dostępności akcji 'set_strong_root_password', która wymaga w ekwipunku 'weak_user_password_detected: true'.
3. Weryfikacja dostępności akcji 'enable_firewall', która wymaga w ekwipunku 'firewall_status_checked: true'.
4. Sprawdzenie flagi available dla tych akcji w stanie początkowym (przed spełnieniem warunków).
5. Próba wykonania niedostępnej akcji przez execute_action i sprawdzenie, czy rzucany jest wyjątek ValueError.

Sukces:
- Akcje, których warunki nie są spełnione, mają atrybut available == False.
- Próba wykonania niedostępnej akcji kończy się błędem (ValueError: Akcja '...' jest obecnie niedostepna.).

Porażka:
- Akcja z niespełnionymi warunkami jest oznaczona jako dostępna (available == True) lub można ją wykonać bez błędu.
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


class TestT4UnavailableActionRequirements(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationMission = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_actions_with_unmet_requirements_are_not_available(self):
        # Akcja 'set_strong_root_password' wymaga weak_user_password_detected: true
        self.assertFalse(
            self.game_logic.actions["set_strong_root_password"].available,
            "Akcja 'set_strong_root_password' nie powinna być dostępna przed wykryciem słabego hasła."
        )

        # Akcja 'enable_firewall' wymaga firewall_status_checked: true
        self.assertFalse(
            self.game_logic.actions["enable_firewall"].available,
            "Akcja 'enable_firewall' nie powinna być dostępna przed sprawdzeniem statusu zapory."
        )

    def test_executing_unavailable_action_raises_value_error(self):
        # Próba wykonania niedostępnej akcji 'set_strong_root_password'
        with self.assertRaises(ValueError) as context:
            self.game_logic.execute_action("set_strong_root_password")
        self.assertIn("jest obecnie niedostepna", str(context.exception))

        # Próba wykonania niedostępnej akcji 'enable_firewall'
        with self.assertRaises(ValueError) as context:
            self.game_logic.execute_action("enable_firewall")
        self.assertIn("jest obecnie niedostepna", str(context.exception))


if __name__ == "__main__":
    unittest.main()
