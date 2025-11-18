import os
import json
from datetime import datetime
from experiments.entities import InvesDetails
from typing import List
from config.paths import *

import dataclasses
import enum
def inves_details_to_dict(obj):
    def remove_boundaries(d):
        if isinstance(d, dict):
            return {k: remove_boundaries(v) for k, v in d.items() if k not in ("boundary_min", "boundary_max")}
        elif isinstance(d, list):
            return [remove_boundaries(i) for i in d]
        else:
            return d
    def convert(val):
        if isinstance(val, enum.Enum):
            return val.value
        elif dataclasses.is_dataclass(val):
            d = dataclasses.asdict(val)
            d = remove_boundaries(d)
            return {k: convert(v) for k, v in d.items()}
        elif isinstance(val, list):
            return [convert(i) for i in val]
        elif isinstance(val, dict):
            return {k: convert(v) for k, v in val.items()}
        else:
            return val
    return convert(obj)
       

def save_inves_details_json(inves_details_list: List[InvesDetails]):
    os.makedirs(PATH_EXP_OUTPUT_DIR, exist_ok=True)
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    filename = f"{PATH_EXP_OUTPUT_DIR}results_{now}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(inves_details_to_dict(inves_details_list), f, indent=2, ensure_ascii=False)
    print(f"Experiment results saved to {filename}")
