"""
Przypadek testowy: T1
Opis: Sprawdź czy zmienna trafiająca do świata gry (worldstate) nie jest dostępna z poziomu ekwipunku (inventory).
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux.yaml'.
2. Odczytanie zmiennej 'firewall_enabled', która jest zdefiniowana w stanie świata gry (worldstate: firewall_enabled: false).
3. Próba odczytania zmiennej 'firewall_enabled' z poziomu ekwipunku gracza (inventory).
4. Modyfikacja stanu świata za pomocą set_worldstate_variable i ponowna weryfikacja izolacji od ekwipunku.

Sukces:
- Zmienna stanu świata 'firewall_enabled' jest poprawnie dostępna w worldstate.
- Zmienna 'firewall_enabled' NIE jest dostępna w ekwipunku (zwraca None oraz brak klucza w inventory).

Porażka:
- Zmienna świata gry jest widoczna/zwracana jako element ekwipunku gracza.
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


class TestT1WorldstateIsolationFromInventory(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationMission = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_worldstate_variable_not_accessible_in_inventory(self):
        # Weryfikacja wartości początkowej w stanie świata
        worldstate_val = self.scenario.get_worldstate_variable("firewall_enabled")
        self.assertIsNotNone(worldstate_val, "Zmienna 'firewall_enabled' powinna istnieć w worldstate.")
        self.assertFalse(worldstate_val, "Wartość początkowa 'firewall_enabled' w worldstate powinna wynosić False.")

        # Weryfikacja braku dostępności zmiennej w ekwipunku
        inventory_val = self.scenario.get_inventory_variable("firewall_enabled")
        self.assertIsNone(inventory_val, "Zmienna ze świata gry nie powinna być dostępna z poziomu ekwipunku (powinna zwracać None).")
        self.assertNotIn("firewall_enabled", self.scenario.inventory, "Klucz 'firewall_enabled' nie powinien istnieć w słowniku inventory.")

    def test_updated_worldstate_variable_remains_isolated_from_inventory(self):
        # Modyfikacja zmiennej w stanie świata
        self.scenario.set_worldstate_variable("firewall_enabled", True)

        # Sprawdzenie czy zmiana jest widoczna w świecie gry
        self.assertTrue(self.scenario.get_worldstate_variable("firewall_enabled"))

        # Sprawdzenie czy nadal nie przeniknęła do ekwipunku
        self.assertIsNone(self.scenario.get_inventory_variable("firewall_enabled"))
        self.assertNotIn("firewall_enabled", self.scenario.inventory)


if __name__ == "__main__":
    unittest.main()

