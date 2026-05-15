# secda_apps_evaluation_suite

This suite generates board-specific run scripts from a JSON experiment description, pushes the generated files to the target board, and then executes the requested apps.

## Current Layout

Hardware configurations are collected from [hardware_automation/configs](../../hardware_automation/configs). The current hardware families in this repository are:

- `ADD`
- `CPU`
- `FCGEMM`
- `SA`
- `VM`

The SECDA app entry points under [src/secda_apps](../secda_apps) are:

- `eval_model`
- `eval_model_accuracy`
- `imagenet_image_classification`

The suite also supports `benchmark_model` and `inference_diff` through the TensorFlow Lite tooling and the generated run scripts.

## How It Works

1. Choose the hardware list from `hardware_automation/configs/*.json`.
2. Choose the app list and arguments in an experiment config under [configs](configs).
3. Use arrays in the JSON when you want the suite to run multiple values in one invocation.
4. Use the `$(d)` token for paths that should be expanded to the board data directory during generation.
5. Run the wrapper script to generate configs, copy files, and launch the experiments.

## Supported App Fields

The example configs in this directory use the following field names.

### `benchmark_model`

- `graph` or `model_file`
- `num_threads` or `threads`
- `num_runs` or `runs`
- Optional fields such as `enable_op_profiling` and `profiling_output_csv_file`

### `inference_diff`

- `model_file` or `tflite_model`
- `num_interpreter_threads` or `threads`
- `num_runs` or `runs`

### `eval_model`

- `tflite_model`
- `labels`
- `image`
- `threads`

### `eval_model_accuracy`

- `tflite_model`
- `threads`
- `labels`
- `image`
- `test_dataset_location`
- `ground_truth_labels_file_name`
- `output_file_name`
- `no_of_images`

### `imagenet_image_classification`

- `model_file`
- `num_interpreter_threads`
- `ground_truth_images_path`
- `ground_truth_labels`
- `model_output_labels`
- `output_file_path`
- `num_images`
- `SkipEvaluation`

## Example Experiment Files

### `configs/default_exp.json`

Minimal example for a single app on two hardware targets:

### `configs/template_exp.json`

Larger example with multiple apps, multiple models, and mixed scalar/list values:


## How To Run

### Prerequisites

Before running any experiments, you **must** complete the board setup steps in [FPGA Setup and Support Documentation](../../docs/fpga_support.md#board-setup-for-secda-apps-evaluation-suite). This includes:
- Configuring SSH key-based authentication
- Enabling passwordless sudo
- Setting up XRT and PYNQ environments

Run the wrapper from the `src/secda_apps_evaluation_suite` directory.

```bash
./secda_apps_evaluation_suite.sh -j configs/default_exp.json
```

Common options:

- `-j <file>` select the experiment JSON file.
- `-i` initialize the board side directories and sync helper scripts.
- `-b` generate binaries.
- `-c` copy bitstreams to the boards.
- `-l` skip loading bitstreams on the board.
- `-p` collect power while running benchmark jobs.
- `-n <name>` name the output folder under `results/`.


### Example 1: Full run on the default experiment

```bash
./secda_apps_evaluation_suite.sh -j configs/default_exp.json -i -b -c -n default_full_run
```

### Example 2: Run benchmark with power collection

```bash
./secda_apps_evaluation_suite.sh -j configs/default_exp.json -b -c -p -n bm_power
```

**Note:** Power collection (with `-p`) only works for KRIA boards. For Z1 and Z2, use a USB power meter for manual measurement.

## Output

- Generated files are written under the experiment output directory configured in `config.json`.
- A copy of the generated results and the input JSON is stored under `results/<name>/`.
- The main log file is `process_flags_n_config.log` in the generated output directory.

## Notes

- Use `$(d)` in paths when the same experiment should resolve correctly on the host and the board.
- Put multiple candidate values in arrays when you want the suite to generate one run per combination.


