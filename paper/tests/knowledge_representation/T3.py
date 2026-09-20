"""
Przypadek testowy: T3
Opis: Sprawdź czy rezultat akcji wykonuje się poprawnie - czy zmiany nim spowodowane zostają nałożone w przypadku jego wystąpienia.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Sprawdzenie stanu początkowego (brak zmiennej 'weak_user_password_detected' w ekwipunku).
3. Wykonanie akcji 'check_user_password' przez grę (game_logic.execute_action).
4. Weryfikacja czy stan gry po wykonaniu akcji został zaktualizowany zgodnie z sekcją 'produces' akcji:
   - Zmienna 'weak_user_password_detected' została ustawiona na True w inventory.
5. Wykonanie kolejnej akcji 'set_strong_root_password' i sprawdzenie czy zmiany w zakresie 'inventory_and_worldstate' ('changed_root_password: true') zostały nałożone zarówno na inventory jak i worldstate.

Sukces:
- Po wykonaniu akcji wszystkie zdefiniowane w niej zmiany stanu (inventory, worldstate, inventory_and_worldstate) zostały poprawnie zaaplikowane w scenariuszu.

Porażka:
- Stan gry po wykonaniu akcji nie uległ zmianie lub nałożono błędne wartości.
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


class TestT3ActionExecutionAndStateChanges(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationScenario = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_action_produces_inventory_change(self):
        # Stan przed wykonaniem akcji
        self.assertIsNone(self.scenario.get_inventory_variable("weak_user_password_detected"))

        # Wykonanie akcji check_user_password
        action = self.game_logic.execute_action("check_user_password")

        # Weryfikacja zwróconej akcji
        self.assertEqual(action.user_input, "echo /root/root_password.txt")

        # Weryfikacja nałożenia zmian w ekwipunku
        self.assertTrue(
            self.scenario.get_inventory_variable("weak_user_password_detected"),
            "Zmienna 'weak_user_password_detected' powinna mieć wartość True po wykonaniu akcji."
        )

    def test_action_produces_inventory_and_worldstate_changes(self):
        # Przygotowanie stanu (wykonanie pierwszej akcji)
        self.game_logic.execute_action("check_user_password")

        # Przed wykonaniem set_strong_root_password
        self.assertIsNone(self.scenario.get_inventory_variable("changed_root_password"))
        self.assertIsNone(self.scenario.get_worldstate_variable("changed_root_password"))

        # Wykonanie akcji set_strong_root_password
        self.game_logic.execute_action("set_strong_root_password")

        # Weryfikacja nałożenia zmian w obu zakresach
        self.assertTrue(
            self.scenario.get_inventory_variable("changed_root_password"),
            "Zmienna 'changed_root_password' powinna zostać zaktualizowana w ekwipunku."
        )
        self.assertTrue(
            self.scenario.get_worldstate_variable("changed_root_password"),
            "Zmienna 'changed_root_password' powinna zostać zaktualizowana w stanie świata."
        )


if __name__ == "__main__":
    unittest.main()
