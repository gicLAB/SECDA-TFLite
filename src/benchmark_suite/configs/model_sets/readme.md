# Model Sets Configuration Files

This directory contains configuration files for different sets of models used in experiments. Each file defines a set of models and their associated parameters, such as layer types. The configuration files needs to be in JSON format and follow the specified schema below.
These files are used to specify which models to include in a particular experiment and the type of layers they contain.

These model sets, if properly configured, will show up in the [secda_benchmarking_suite.ipynb](../../secda_benchmarking_suite.ipynb) notebook under the "Model Sets" section. This allows users to easily select and run experiments with custom predefined sets of models. 

Model set simply represents the name of the model files, these files need exist in the [model_gen/models](../../model_gen/models) directory. The model files should be in the format `<model_name>.tflite`, where `<model_name>` corresponds to the names specified in the configuration files.


## JSON Schema for model_sets configuration files

The configuration file `tconv_exp.json` follows this schema:

```
{
  "<experiment_name>": {
    "models": [
      "<model_name_1>",
      "<model_name_2>",
      "..."
    ],
    "layer_type": "<LAYER_TYPE>"
  }
}
```

- `<experiment_name>`: (string) The name of the experiment (e.g., `exp_name`).
- `models`: (array of strings) List of model or layer names included in the experiment.
- `layer_type`: (string) The type of layer for all models in this experiment (e.g., `TRANSPOSE_CONV`).

**Example:**
```
{
    "tconv_exp": {
        "models": [
            "dcgan_layer1",
            "dcgan_layer2",
            "dcgan_layer3",
            "dcgan_layer4",
            "fcn_layer1",
            "fcn_layer2",
            "fcn_layer3",
            "style_transfer_layer1",
            "style_transfer_layer2",
            "style_transfer_layer3",
            "fsrcnn_layer1"
        ],
        "layer_type": "TRANSPOSE_CONV"
    }
}