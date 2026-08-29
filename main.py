import yaml
from parser.parser import MissionRepresentation
    

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

mission = load_yaml("data/missions/mission_01.yaml")
print(yaml.dump(mission, sort_keys=False, allow_unicode=True))

mission_representation = MissionRepresentation(mission)

print(mission_representation.mission_meta.get("title"))
print(mission_representation.mission_meta)
print(mission_representation.inventory.get("title"))
print(mission_representation.actions.keys())