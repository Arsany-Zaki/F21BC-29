import os
import json
from datetime import datetime
from experiments.entities import InvesDetails
from typing import List
from config.paths import *
import shutil
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
       
def log_config_and_result(inves_details_list: List[InvesDetails]):
    save_result(inves_details_list)
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    archive_config(now)
    archive_result(now)
    
def save_result(inves_details_list: List[InvesDetails]):
    with open(PATH_EXP_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(inves_details_to_dict(inves_details_list), f, indent=2, ensure_ascii=False)
    print(f"Experiment results saved to {PATH_EXP_RESULT_FILE}")

def archive_config(timestamp: str):
    os.makedirs(PATH_EXP_CONFIG_ARCHIVE_DIR, exist_ok=True)
    orig_file_path = PATH_EXP_CONFIG_FILE
    archive_file_path = f"{PATH_EXP_CONFIG_ARCHIVE_DIR}config_{timestamp}.yaml"
    shutil.copyfile(orig_file_path, archive_file_path)

def archive_result(timestamp: str):
    os.makedirs(PATH_EXP_RESULT_ARCHIVE_DIR, exist_ok=True)
    orig_file_path = PATH_EXP_RESULT_FILE
    archive_file_path = f"{PATH_EXP_RESULT_ARCHIVE_DIR}result_{timestamp}.json"
    shutil.copyfile(orig_file_path, archive_file_path)