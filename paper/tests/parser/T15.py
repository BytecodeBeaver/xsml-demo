"""
Przypadek testowy: T15
Opis: Walidator XSML powinien odrzucić scenariusz, w którym przejście (transition) odwołuje się do nieistniejącej akcji (action_trigger).
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku 'hardening_linux.yaml' i utworzenie jego kopii do modyfikacji.
2. Podmiana pola 'action_trigger' w pierwszym przejściu scenariusza na identyfikator akcji, który nie istnieje w sekcji 'actions'.
3. Wywołanie XSMLParser.parse na zmodyfikowanych danych.

Sukces:
- Parser zgłasza wyjątek ValueError informujący o wskazaniu nieistniejącej akcji.

Porażka:
- Parser akceptuje scenariusz mimo odwołania do nieistniejącej akcji.
"""

import copy
import os
import sys
import unittest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from parser.parser import XSMLParser

MISSION_PATH = os.path.join(PROJECT_ROOT, "data", "missions", "hardening_linux.yaml")


class TestT15TransitionToNonexistentAction(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            self.yaml_data = yaml.safe_load(file)

    def test_transition_action_trigger_nonexistent_raises_value_error(self):
        data = copy.deepcopy(self.yaml_data)
        scenario = data["mission"]["scenarios"][0]
        scenario["transitions"][0]["action_trigger"] = "nonexistent_action"

        with self.assertRaises(ValueError) as context:
            XSMLParser.parse(data)

        self.assertIn("nieistniejaca akcje", str(context.exception))


if __name__ == "__main__":
    unittest.main()
