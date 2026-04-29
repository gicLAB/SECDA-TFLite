import json
import sys
import os
import shutil

# Get the actual workspace path (parent directory of scripts)
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

src_config_path = os.path.join(workspace_path, "config.json")
with open(src_config_path, "r") as file:
    config = json.load(file)

config["vivado_2019_path"] = ""
config["vivado_2024_path"] = ""
config["secda_tflite_path"] = workspace_path

config["models_dirs"] = [
    f"{workspace_path}/data/models",
    f"{workspace_path}/src/benchmark_suite/model_gen/models",
    f"{workspace_path}/tensorflow/models",
]


# Attempt to create a symlink from .devcontainer/config.json -> ../config.json
# If the modified config differs from the source config, write the modified
# config into .devcontainer/config.json instead. Use --force-symlink to
# override and force a symlink (this will make the devcontainer file point
# to the top-level config.json without applying the modified values).
devcontainer_dir = os.path.join(workspace_path, ".devcontainer")
devcontainer_cfg = os.path.join(devcontainer_dir, "config.json")
os.makedirs(devcontainer_dir, exist_ok=True)

force_symlink = "--force-symlink" in sys.argv

print(f"Force symlink: {force_symlink}")

try:
    # Reload the original top-level config to compare against our modified one
    with open(src_config_path, "r") as f:
        original = json.load(f)
except Exception:
    original = None

def safe_remove(path):
    try:
        if os.path.islink(path) or os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

if force_symlink:
    safe_remove(devcontainer_cfg)
    try:
        target = os.path.relpath(src_config_path, devcontainer_dir)
        os.symlink(target, devcontainer_cfg)
        print(f"Created symlink: {devcontainer_cfg} -> {target}")
    except Exception as e:
        # Fallback to writing the modified config if symlink creation fails
        with open(devcontainer_cfg, "w") as file:
            json.dump(config, file, indent=2)
        print(f"Symlink failed ({e}); wrote file instead: {devcontainer_cfg}")
else:
    # If our modified config matches the source, we can safely symlink.
    if original is not None and config == original:
        safe_remove(devcontainer_cfg)
        try:
            target = os.path.relpath(src_config_path, devcontainer_dir)
            os.symlink(target, devcontainer_cfg)
            print(f"Created symlink: {devcontainer_cfg} -> {target}")
        except Exception as e:
            with open(devcontainer_cfg, "w") as file:
                json.dump(config, file, indent=2)
            print(f"Symlink failed ({e}); wrote file instead: {devcontainer_cfg}")
    else:
        # Configs differ (or we couldn't read original) — write the modified config
        with open(devcontainer_cfg, "w") as file:
            json.dump(config, file, indent=2)
        print(f"Configuration updated successfully and saved to {devcontainer_cfg}")