"""
Przypadek testowy: T14
Opis: Walidator XSML powinien odrzucić scenariusz, w którym przejście (transition) wskazuje na nieistniejący węzeł (node).
Scenariusz bazowy: data/missions/hardening_linux.yaml

Kroki testowe:
1. Wczytanie pliku 'hardening_linux.yaml' i utworzenie jego kopii do modyfikacji.
2. Podmiana pola 'to' w pierwszym przejściu scenariusza 'linux_hardening_walkthrough' na identyfikator węzła, który nie istnieje w sekcji 'nodes'.
3. Wywołanie XSMLParser.parse na zmodyfikowanych danych.

Sukces:
- Parser zgłasza wyjątek ValueError informujący o wskazaniu nieistniejącego węzła.

Porażka:
- Parser akceptuje scenariusz mimo odwołania do nieistniejącego węzła.
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


class TestT14TransitionToNonexistentNode(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            self.yaml_data = yaml.safe_load(file)

    def test_transition_to_nonexistent_node_raises_value_error(self):
        data = copy.deepcopy(self.yaml_data)
        scenario = data["mission"]["scenarios"][0]
        scenario["transitions"][0]["to"] = "nonexistent_node"

        with self.assertRaises(ValueError) as context:
            XSMLParser.parse(data)

        self.assertIn("nieistniejacy wezel", str(context.exception))


if __name__ == "__main__":
    unittest.main()
