"""
Read a json file and convert all the key and value to environment variables.
"""

import json
import os
from dotenv import load_dotenv


def load_env(file_path: str, env_prefix: str = "") -> None:
    # Load the dotenv file
    load_dotenv()
    # Read the json file
    with open(file_path, "r") as f:
        data = json.load(f)
    # Convert the data to environment variables
    for key, value in data.items():
        if env_prefix:
            os.environ[f"{env_prefix}_{key}"] = str(value)
        else:
            os.environ[key] = str(value)
