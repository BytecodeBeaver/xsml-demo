"""
Przypadek testowy: T17
Opis: Walidator XSML powinien odrzucić akcję, której suma prawdopodobieństw wszystkich rezultatów (produces) nie wynosi 1.0.
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku 'hardening_linux.yaml' i utworzenie jego kopii do modyfikacji.
2. Podmiana wartości 'probability' jedynego rezultatu akcji 'check_firewall_status' na 0.5 (suma przestaje wynosić 1.0).
3. Wywołanie XSMLParser.parse na zmodyfikowanych danych.

Sukces:
- Parser zgłasza wyjątek ValueError informujący, że suma prawdopodobieństw rezultatów akcji musi wynosić 1.0.

Porażka:
- Parser akceptuje akcję, dla której suma prawdopodobieństw rezultatów jest różna od 1.0.
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


class TestT17ProbabilitiesDoNotSumToOne(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            self.yaml_data = yaml.safe_load(file)

    def test_probabilities_not_summing_to_one_raises_value_error(self):
        data = copy.deepcopy(self.yaml_data)
        actions = data["mission"]["actions"]
        action = next(a for a in actions if a["id"] == "check_firewall_status")
        action["produces"][0]["probability"] = 0.5

        with self.assertRaises(ValueError) as context:
            XSMLParser.parse(data)

        self.assertIn("musza sumowac sie do 1.0", str(context.exception))


if __name__ == "__main__":
    unittest.main()
