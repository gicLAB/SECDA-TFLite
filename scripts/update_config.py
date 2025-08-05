import json
import sys
import os   

# Get the actual workspace path (parent directory of scripts)
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open("../config.json", "r") as file:
    config = json.load(file)

config["vivado_2019_path"] = ""
config["vivado_2024_path"] = ""
config["secda_tflite_path"] = workspace_path

config["models_dirs"] = [
    f"{workspace_path}/data/models",
    f"{workspace_path}/src/benchmark_suite/model_gen/models",
    f"{workspace_path}/tensorflow/models"
]



with open("../.devcontainer/config.json", "w") as file:
    json.dump(config, file, indent=2)


print("Configuration updated successfully and saved to ../.devcontainer/config.json")