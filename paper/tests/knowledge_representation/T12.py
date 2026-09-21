"""
Przypadek testowy: T12
Opis: Sprawdź, czy gracz może zobaczyć listę możliwych scenariuszy / proponowanych przebiegów rozgrywki na poziomie logiki gry.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku YAML misji.
2. Parsowanie danych przez XSMLParser.parse.
3. Wygenerowanie listy przewodników wykonania wywołaniem build_walkthroughs.
4. Weryfikacja, że w liście znajduje się scenariusz 'linux_hardening_walkthrough'.

Sukces:
- build_walkthroughs zwraca co najmniej jeden scenariusz.
- Wśród wyników znajduje się scenariusz 'linux_hardening_walkthrough'.
- Każdy element zawiera identyfikator i opis celu oraz listę kroków.

Porażka:
- Lista scenariuszy jest pusta albo nie zawiera oczekiwanego scenariusza.
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


class TestT12PlayerCanSeeAvailableScenarios(unittest.TestCase):
    def test_player_receives_list_of_available_scenarios(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        scenario = XSMLParser.parse(yaml_data)
        walkthroughs = build_walkthroughs(scenario)

        self.assertIsInstance(walkthroughs, list)
        self.assertGreater(len(walkthroughs), 0)

        scenario_ids = [item["id"] for item in walkthroughs]
        self.assertIn("linux_hardening_walkthrough", scenario_ids)

        walkthrough = next(item for item in walkthroughs if item["id"] == "linux_hardening_walkthrough")
        self.assertIn("goal", walkthrough)
        self.assertIn("goal_description", walkthrough)
        self.assertIn("steps", walkthrough)
        self.assertIsInstance(walkthrough["steps"], list)


if __name__ == "__main__":
    unittest.main()
