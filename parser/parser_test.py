import yaml
from parser import SimulationScenario, XSMLParser


# def print_yaml_types(value, path="root"):
#     print(f"{path}: {type(value).__name__} = {value!r}")

#     if isinstance(value, dict):
#         for key, nested_value in value.items():
#             print_yaml_types(nested_value, f"{path}.{key}")
#     elif isinstance(value, list):
#         for index, nested_value in enumerate(value):
#             print_yaml_types(nested_value, f"{path}[{index}]")

# with open("data/missions/hardening_linux_no_comment.yaml", "r") as file:
#     yaml_data = yaml.safe_load(file)
#     print_yaml_types(yaml_data)
#     scenario: SimulationScenario = XSMLParser.parse(yaml_data)

with open("data/missions/hardening_linux_no_comment.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)
    scenario: SimulationScenario = XSMLParser.parse(yaml_data)
    print(scenario.environment["firewall_enabled"])