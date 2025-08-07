#!/bin/bash

# Function to search and replace a string in a file
replace_string_in_file() {
  local file="$1"
  local search="$2"
  local replace="$3"
  sed -i "s|${search}|${replace}|g" "$file"
}



vivado_2019_path=$(jq -r '.vivado_2019_path' ../config.json)
vivado_2024_path=$(jq -r '.vivado_2024_path' ../config.json)

# Remove the last occurrence of '2024.1/bin/' from vivado_2024_path if present
vivado_2024_path="${vivado_2024_path%/2024.1/bin/}"
# Remove the last occurrence of '2019.2/bin/' from vivado_2019_path if present
vivado_2019_path="${vivado_2019_path%/2019.2/bin/}"


# Example usage: search and replace in devcontainer.json
replace_string_in_file "../.devcontainer/devcontainer.json" "VIVADO_2019_PATH" $vivado_2019_path
replace_string_in_file "../.devcontainer/devcontainer.json" "VIVADO_2024_PATH" $vivado_2024_path


pushd ../tensorflow/tensorflow/lite/examples/
ln -s -f ../../../../src/secda_apps/ ./ 
popd
pushd ../tensorflow/tensorflow/lite/delegates/utils/
ln -s -f ../../../../../src/secda_delegates/ ./
ln -s -f ../../../../../src/.clang-format ./
popd