# Benchmark Suite

## Purpose
The Benchmark Suite is designed to evaluate the performance of various models on SECDA-TFLite accelerators.
It provides graphical interface to set up experiments, run benchmarks, and visualize results.

## Getting Started

1. Open [secda_benchmarking_suite.ipynb](./secda_benchmarking_suite.ipynb) in Jupyter Notebook.
2. Execute the cell
3. Configure options to set up the experiment.
4. Run the experiment.


## Selection Interfaces

We explain the options available in the Benchmark Suite in more detail below.

### Model Selection

- We provide multiple list of models to choose from to run the benchmark on.
- For example, the "ADD" list contains models that always contain the ADD layers
- These model lists are grouped by the type of operations that is listed in the [model config file](./configs/models.json) file.
- Using [model config file](./configs/models.json) file, you can add your own models to the list but make sure you add your model file in the [models](./model_gen/models/) directory.

- You can add custom sets of models by creating a new JSON file in the [configs/model_sets](./configs/model_sets/) directory.
- The model sets will be automatically detected and listed in the "Model Sets" section of the notebook.
- Each model set should follow the JSON schema defined in the [model_sets/readme.md](./configs/model_sets/readme.md) file.



## Layer Selection
- The "Layer Selection" section allows you to choose the type of layers to benchmark.
- The layers chosen here will be grouped as "acc_layers" which stands for accelerated or target layers.
- For CPU benchmarks, the "acc_layers" will be used to filter the layers that we want to compare against an accelerator benchmark.
- For example, if you select "ADD" layers, and run an accelerator that supports ADD layers, the benchmark will group the results based on the ADD layers and non-ADD layers. If we run a CPU benchmark, it will group the layers similarly, so it is easy to compare the performance benefit of the accelerator against the CPU.

## Accelerator Selection
- The "Accelerator Selection" section allows you to choose the accelerators to benchmark against.
- The accelerators are listed based on the configuration files defined in the [hardware_automation configuration directory](../../hardware_automation/configs/)

- Note the bitstreams for the accelerator are not automatically generated, you need to run the [hardware automation](../../hardware_automation/) to generate the bitstreams for the accelerators. Check the [hardware automation readme](../../hardware_automation/readme.md) for more details on how to generate the bitstreams.
- The bitstreams are expected to be stored in either [bitstreams directory](./bitstreams/) or  already inside the target board's "board_dir/bitstreams" folder (defined in your [global config file](../../config.json)).

- Note: to compare against the CPU, you need to select the target CPU from the CPU list in the "Accelerator Selection" section. We currently device CPUs:
  - PYNQ Z1/Z2 : `CPU`
  - KV260 Kria : `CPU_KRIA`


## Experiment Configuration Interface

- The "Experiment Configuration" section allows you to configure the experiment you want to run.
- There are many options to configure the experiment:
  - **Experiment Name**: This is the name of the experiment. It will be used to create a directory for the experiment results. The directory will be created in the `tmp` directory.

  - **Number of Runs**: This is the number of times you want to run the experiment. The results will be averaged over the number of runs. This is useful to get a more accurate measurement of the performance.

  - **Threads**: This is the number of threads to use for the experiment. This is useful if you want to run the experiment on multiple threads configurations. For example, if you want to run the experiment on 1 thread and 2 threads, you can set the threads to `1,2` and the experiment will run on all three configurations.

  
  - **Initialize boards**: This option will initialize the boards before running the experiment. This is useful if you want to setup your FPGA device with the correct directory structure and scripts for the secda-tflite benchmarking suite. It will also copy the [models](./model_gen/models/) directory and the [bitstreams](./bitstreams/) directory to the target board/s.
  - 
  - **Send Models**: This option will send the selected models to the target board/s. Use this if you have updated the models in the [models](./model_gen/models/) directory and want to send the updated models to the target board/s.

  - **Skip running experiment**: This option will skip running the experiment and instead only try to process the results from the previous experiment. This is useful if you want to see the data from the previous experiment without running the experiment again.
  
  - **Skip inference difference checks**: This option will skip the inference difference checks. This is useful if you don't want to validate the accelerator results against the CPU results and only care about the performance of the accelerator.
  
  - **Generate binaries**: This option will generate the binaries for the accelerator delegates. If simulation mode is not selected then this will cross compile and send the binaries to the target board/s. If simulation mode is selected, it will generate the binaries for the host machine.

  - **Generate run scripts**: This option will generate the run scripts for the experiment. This is useful if you want to run the experiment manually on the target board/s. The run scripts will be generated at [run_exp.sh](./generated/run_exp.sh). **We highly recommend** you to run the experiments using the generated run scripts instead of running the experiment from the Jupyter Notebook. This is because the Jupyter Notebook may not be able to handle large amount of data and may crash or hang. To run the experiment manually: ``` ./generated/run_exp.sh``` within the [benchmark_suite](./) directory.
  
  - **Simulation Mode**: This option will run the experiment in simulation mode. This is will run the experiment on the host machine instead of the target board/s. This is useful if you want to test the experiment without running it on the target board/s. **We highly recommend** you to run one model+accelerator combination at a time in simulation mode, as it will take a couple of minutes to run the experiment for each model+accelerator combination.

  - **Test Run**: This option will run a test run of the experiment. This is useful if you want to quickly test a model+accelerator combination without running the full experiment or benchmarking performance. It will run the model on the accelerator and print  verification results.

  - **Timeout (s)**: This option will set the timeout for the experiment. Each run (model+accelerator) can take no longer than the specified time, it will be terminated. This is useful if you want to avoid long-running experiments that may hang or crash.

  - **Load Widget Values**: This button will load the widget values from the [configs/widget_values.json](./configs/widget_values.json) file. This is useful if you want to quickly load the previously saved experiment configuration.
  
  - **Save Widget Values**: This button will save the widget values to the [configs/widget_values.json](./configs/widget_values.json) file. This is useful if you want to save the current experiment configuration for later use.


## Known Issues
- Sometimes the Jupyter Notebook may output logs multiple times, due to ipywidget bug, to fix this restart the Jupyter Notebook server, you might have to switch python kernel back and forth to properly reset the ipywidgets