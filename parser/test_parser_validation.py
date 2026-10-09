import copy
import unittest
from pathlib import Path

import yaml

from parser.parser import XSMLParser


class ParserValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mission_path = Path(__file__).parents[1] / "data" / "missions" / "hardening_linux.yaml"
        with mission_path.open(encoding="utf-8") as file:
            cls.mission_data = yaml.safe_load(file)

    def assert_invalid_mission(self, mutate):
        mission_data = copy.deepcopy(self.mission_data)
        mutate(mission_data)
        with self.assertRaises(ValueError):
            XSMLParser.parse(mission_data)

    def test_transition_to_unknown_node_is_rejected(self):
        self.assert_invalid_mission(
            lambda data: data["mission"]["scenarios"][0]["transitions"][0].__setitem__(
                "to", "unknown_node"
            )
        )

    def test_transition_with_unknown_action_is_rejected(self):
        self.assert_invalid_mission(
            lambda data: data["mission"]["scenarios"][0]["transitions"][0].__setitem__(
                "action_trigger", "unknown_action"
            )
        )

    def test_unknown_scenario_start_node_is_rejected(self):
        self.assert_invalid_mission(
            lambda data: data["mission"]["scenarios"][0].__setitem__("start", "unknown_node")
        )

    def test_unknown_scenario_final_node_is_rejected(self):
        self.assert_invalid_mission(
            lambda data: data["mission"]["scenarios"][0].__setitem__("final", "unknown_node")
        )

    def test_scenario_requires_start_and_final_nodes(self):
        self.assert_invalid_mission(
            lambda data: data["mission"]["scenarios"][0].pop("start")
        )
        self.assert_invalid_mission(
            lambda data: data["mission"]["scenarios"][0].pop("final")
        )

    def test_probability_above_one_is_rejected(self):
        self.assert_invalid_mission(
            lambda data: data["mission"]["actions"][0]["produces"][0].__setitem__(
                "probability", 1.4
            )
        )


if __name__ == "__main__":
    unittest.main()
