import json
from typing import Tuple

def read_config(path : str) -> 'Tuple[dict, dict, dict]':
    f = open(path)
    config = json.load(f)
    f.close()
    return config["evolution"], config["individual"], config["graph"], config["neat"]

