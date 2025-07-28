import json
import sys
import os   

with open("../config.json", "r") as file:
    config = json.load(file)

config["vivado_2019_path"] = ""
config["vivado_2024_path"] = ""
config["secda_tflite_path"] = "/working_dir/SECDA-TFLite/"

config["models_dirs"] = [
    "/working_dir/SECDA-TFLite/data/models",
    "/working_dir/SECDA-TFLite/src/benchmark_suite/model_gen/models",
    "/working_dir/SECDA-TFLite/tensorflow/models"
] + config["models_dirs"]



with open("../.devcontainer/config.json", "w") as file:
    json.dump(config, file, indent=2)


print("Configuration updated successfully and saved to ../.devcontainer/config.json")