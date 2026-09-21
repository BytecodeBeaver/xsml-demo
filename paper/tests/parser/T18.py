"""
Przypadek testowy: T18
Opis: Walidator XSML powinien odrzucić misję, w której występują zduplikowane identyfikatory akcji.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku 'hardening_linux.yaml' i utworzenie jego kopii do modyfikacji.
2. Zmiana identyfikatora drugiej akcji na taki sam jak pierwszej akcji.
3. Wywołanie XSMLParser.parse na zmodyfikowanych danych.

Sukces:
- Parser zgłasza wyjątek ValueError informujący, że identyfikatory akcji muszą być unikalne.

Porażka:
- Parser akceptuje misję, w której występują duplikaty identyfikatorów akcji.
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


class TestT18DuplicateActionIds(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            self.yaml_data = yaml.safe_load(file)

    def test_duplicate_action_ids_raise_value_error(self):
        data = copy.deepcopy(self.yaml_data)
        actions = data["mission"]["actions"]
        actions[1]["id"] = actions[0]["id"]

        with self.assertRaises(ValueError) as context:
            XSMLParser.parse(data)

        self.assertIn("Identyfikatory akcji", str(context.exception))


if __name__ == "__main__":
    unittest.main()
