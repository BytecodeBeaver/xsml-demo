"""
Przypadek testowy: T16
Opis: Walidator XSML powinien odrzucić akcję, której rezultat (produces) ma prawdopodobieństwo spoza zakresu [0.0, 1.0], np. 1.4.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku 'hardening_linux.yaml' i utworzenie jego kopii do modyfikacji.
2. Podmiana wartości 'probability' pierwszego rezultatu akcji 'check_user_password' na 1.4.
3. Wywołanie XSMLParser.parse na zmodyfikowanych danych.

Sukces:
- Parser zgłasza wyjątek ValueError informujący o niepoprawnym prawdopodobieństwie rezultatu.

Porażka:
- Parser akceptuje rezultat akcji o prawdopodobieństwie spoza zakresu [0.0, 1.0].
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


class TestT16ProbabilityOutOfRange(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            self.yaml_data = yaml.safe_load(file)

    def test_probability_above_one_raises_value_error(self):
        data = copy.deepcopy(self.yaml_data)
        actions = data["mission"]["actions"]
        action = next(a for a in actions if a["id"] == "check_user_password")
        action["produces"][0]["probability"] = 1.4

        with self.assertRaises(ValueError) as context:
            XSMLParser.parse(data)

        self.assertIn("Prawdopodobienstwo", str(context.exception))


if __name__ == "__main__":
    unittest.main()
