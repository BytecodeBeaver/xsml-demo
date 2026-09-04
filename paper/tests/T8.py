"""
Przypadek testowy: T8
Opis: Cel główny powinien być uznany za nieosiągnięty, jeśli zarówno warunki osiągnięcia celu głównego jak i cele szczegółowe nie są osiągnięte.
Scenariusz bazowy: data/missions/hardening_linux_no_comment.yaml

Kroki testowe:
1. Wczytanie scenariusza 'hardening_linux_no_comment.yaml'.
2. Skonfigurowanie w celu głównym 'complete_hardening_of_linux_system' dodatkowego niespełnionego wymagania (np. requires: inventory: {'system_audit_approved': True}).
3. Pozostawienie podcelów w stanie nieosiągniętym (brak wykonanych akcji).
4. Aktualizacja i sprawdzenie stanu celów w GameLogic.
5. Weryfikacja czy zarówno warunki własne, jak i podcele oraz cel główny mają status nieosiągnięty/niespełniony.

Sukces:
- Warunki własne celu głównego NIE są spełnione (are_requirements_fullfilled zwraca False).
- Cele szczegółowe NIE są osiągnięte (subgoals achieved == False).
- Cel główny ma status achieved == False.

Porażka:
- Cel główny zostaje oznaczony jako achieved == True pomimo braku spełnienia warunków własnych i podcelów.
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


class TestT8MainGoalUnachievedWhenBothRequirementsAndSubgoalsNotMet(unittest.TestCase):
    def setUp(self):
        with open(MISSION_PATH, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        self.scenario: SimulationScenario = XSMLParser.parse(yaml_data)
        self.game_logic = GameLogic(self.scenario)

    def test_main_goal_unachieved_when_neither_requirements_nor_subgoals_met(self):
        main_goal = self.game_logic.goals["complete_hardening_of_linux_system"]

        # Dodanie do wymagań celu głównego warunku, który nie jest spełniony
        main_goal.requires = {
            "inventory": {"system_audit_approved": True},
            "worldstate": {}
        }
        self.game_logic.update_goal_progress()

        # 1. Sprawdzenie czy warunki własne celu głównego NIE są spełnione
        self.assertFalse(
            self.game_logic.are_requirements_fullfilled(main_goal.requires),
            "Warunki własne celu głównego powinny być niespełnione."
        )

        # 2. Sprawdzenie czy podcele NIE są osiągnięte
        for subgoal in main_goal.subgoals.values():
            self.assertFalse(
                subgoal.achieved,
                f"Podcel '{subgoal.id}' nie powinien być osiągnięty."
            )

        # 3. Sprawdzenie czy cel główny jest nieosiągnięty
        self.assertFalse(
            main_goal.achieved,
            "Cel główny nie powinien być osiągnięty, gdy ani warunki własne, ani podcele nie są spełnione."
        )


if __name__ == "__main__":
    unittest.main()
