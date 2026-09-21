"""
Przypadek testowy: T13
Opis: Sprawdź, czy gracz może zobaczyć zestaw akcji prowadzących do ukończenia gry na podstawie konkretnego scenariusza, w logice gry bez HTTP.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku YAML misji.
2. Parsowanie danych przez XSMLParser.parse.
3. Wygenerowanie listy przewodników wywołaniem build_walkthroughs.
4. Weryfikacja, że w scenariuszu linux_hardening_walkthrough występują etapy oraz listy akcji.

Sukces:
- build_walkthroughs zwraca listę przewodników.
- Wśród nich znajduje się scenariusz 'linux_hardening_walkthrough'.
- Każdy krok listy zawiera opis od-do oraz poprawną listę akcji.

Porażka:
- Brak sekwencji kroków lub akcji w wygenerowanym przewodniku dla scenariusza.
"""

import os
import sys
import unittest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import build_walkthroughs
from parser.parser import XSMLParser

MISSION_PATH = os.path.join(PROJECT_ROOT, "data", "missions", "hardening_linux.yaml")


class TestT13PlayerReceivesActionListFromScenarioWalkthrough(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            self.yaml_data = yaml.safe_load(file)
        self.scenario = XSMLParser.parse(self.yaml_data)

    def test_player_receives_action_list_from_scenario_walkthrough(self):
        walkthroughs = build_walkthroughs(self.scenario)

        self.assertIsInstance(walkthroughs, list)
        self.assertGreater(len(walkthroughs), 0)

        walkthrough = next((item for item in walkthroughs if item["id"] == "linux_hardening_walkthrough"), None)
        self.assertIsNotNone(walkthrough, "Brakuje scenariusza 'linux_hardening_walkthrough' w wygenerowanych przewodnikach.")

        self.assertIn("goal", walkthrough)
        self.assertIn("goal_description", walkthrough)
        self.assertIn("steps", walkthrough)

        self.assertIsInstance(walkthrough["steps"], list)
        self.assertGreater(len(walkthrough["steps"]), 0)

        first_step = walkthrough["steps"][0]
        self.assertIn("from_description", first_step)
        self.assertIn("to_description", first_step)
        self.assertIn("actions", first_step)

        self.assertIsInstance(first_step["actions"], list)
        self.assertGreater(len(first_step["actions"]), 0)

        first_action = first_step["actions"][0]
        self.assertIn("id", first_action)
        self.assertIn("description", first_action)
        self.assertTrue(first_action["id"])
        self.assertTrue(first_action["description"])


if __name__ == "__main__":
    unittest.main()
