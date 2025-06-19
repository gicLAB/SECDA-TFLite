# Benchmark Suite

## Purpose
The Benchmark Suite is designed to evaluate the performance of various models on SECDA-TFLite accelerators.
It provides graphical interface to set up experiments, run benchmarks, and visualize results.

## Getting Started

1. Open secda_benchmarking_suite.ipynb in Jupyter Notebook.
2. Execute the cell
3. Configure options to set up the experiment.
4. Run the experiment.


## Selection Interfaces

We explain the options available in the Benchmark Suite in more detail below.

### Models

- We provide multiple list of models to choose from to run the benchmark on.
- For example, the "ADD" list contains models that always contain the ADD layers
- These model lists are grouped by the type of operations that is listed in the [model config file](./configs/models.json) file.
- Using [model config file](./configs/models.json) file, you can add your own models to the list but make sure you add your model file in the [models](./model_gen/models/) directory.
- 