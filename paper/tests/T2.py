"""
Przypadek testowy: T2
Opis: Sprawdź czy zmienna trafiająca do ekwipunku (inventory) jest dostępna z poziomu ekwipunku.
Scenariusz bazowy: data/missions/hardening_linux_no_comment.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux_no_comment.yaml'.
2. Dodanie zmiennej do ekwipunku gracza (np. 'weak_user_password_detected: true' poprzez set_inventory_variable lub wykonanie akcji 'check_user_password').
3. Odczytanie zmiennej za pomocą metody get_inventory_variable oraz bezpośrednio ze słownika inventory.
4. Weryfikacja czy zmienna ma poprawną wartość i jest dostępna w ekwipunku.

Sukces:
- Zmienna dodana do ekwipunku jest poprawnie odczytywana z poziomu ekwipunku (get_inventory_variable zwraca True).
- Zmienna znajduje się w słowniku ekwipunku scenario.inventory.

Porażka:
- Zmienna trafiająca do ekwipunku nie jest dostępna z poziomu ekwipunku (zwraca None lub brak klucza).
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

MISSION_PATH = os.path.join(PROJECT_ROOT, "data", "missions", "hardening_linux_no_comment.yaml")


class TestT2InventoryVariableAccessibility(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationScenario = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_variable_added_to_inventory_is_accessible_via_get_inventory_variable(self):
        # Stan początkowy - brak zmiennej w ekwipunku
        self.assertIsNone(self.scenario.get_inventory_variable("weak_user_password_detected"))

        # Wprowadzenie zmiennej do ekwipunku
        self.scenario.set_inventory_variable("weak_user_password_detected", True)

        # Weryfikacja dostępności zmiennej z poziomu ekwipunku
        val = self.scenario.get_inventory_variable("weak_user_password_detected")
        self.assertIsNotNone(val, "Zmienna powinna być dostępna w ekwipunku.")
        self.assertTrue(val, "Zmienna w ekwipunku powinna mieć wartość True.")
        self.assertIn("weak_user_password_detected", self.scenario.inventory)
        self.assertEqual(self.scenario.inventory["weak_user_password_detected"], True)

    def test_variable_produced_by_action_is_accessible_in_inventory(self):
        # Wykonanie akcji generującej wpis w ekwipunku
        self.game_logic.execute_action("check_user_password")

        # Weryfikacja obecności zmiennej w ekwipunku
        inventory_val = self.scenario.get_inventory_variable("weak_user_password_detected")
        self.assertTrue(inventory_val, "Zmienna 'weak_user_password_detected' wygenerowana przez akcję powinna być dostępna w inventory.")


if __name__ == "__main__":
    unittest.main()



