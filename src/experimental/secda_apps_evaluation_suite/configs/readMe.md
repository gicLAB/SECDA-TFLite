
## How should we configure the benchmark suite?

1. We have different apps and different hardware.

2. In each app, we have different arguments.

3. Within these different arguments, might want to run some arguments with different variations in a single run.

4. Hardware names should be collected from the hardware_automation/configs/*.json files.

5. Collect apps name from the src/secda_apps directory.

6. Benchmark_model supported argument variations:
    - go to the model_n_data dir and copy the model path in the "models" argument witoout the extension ".tflite"
    - "models" : ["model1", "model2", "model3"] // multiple values supported
    - "models" : "model_name.json" // json file with multiple values
    - "threads": ["1","2"], // only multiple values accepted
    - "runs"   : "100" // only single value accepted

7. inference_diff supported argument variations:
    - go to the model_n_data dir and copy the model path in the "models" argument witoout the extension ".tflite"
    - "models" : ["model1", "model2", "model3"] // multiple values supported
    - "models" : "model_name.json" // json file with multiple values
    - "threads": ["1","2"], // only multiple values accepted
    - "runs"   : "100" // only single value accepted

8. em_apps supported argument variations:
    - currently apps only support for imageNet models, so model should be in the dir (/mnt/sata/model_n_data/imagenet/models)
    - make sure given image name is available in data_dir (/mnt/sata/model_n_data/imagenet/otherImages)
    - "models" : ["model1", "model2", "model3"] // multiple values supported
    - "threads": ["1","2"], // only multiple values accepted

9. ema_apps supported argument variations:
    - go to the model_n_data dir and copy the model path in the "models" argument witoout the extension ".tflite"
    - currently supports only cifar 10 models, so model should be in the dir (/mnt/sata/model_n_data/cifar10/models)
    - "models" : ["model1", "model2", "model3"] // multiple values supported
    - "threads": ["1","2"], // only multiple values accepted
    - "img_no"  : "10000" // only single value accepted(highest value is 10000)

10. iic_apps supported argument variations:
    - go to the model_n_data dir and copy the model path in the "models" argument witoout the extension ".tflite"
    - currently apps only support for imageNet models, so model should be in the dir (/mnt/sata/model_n_data/imagenet/models)
    - "models" : ["model1", "model2", "model3"] // multiple values supported
    - "threads": ["1","2"], // only multiple values accepted
    - "img_no"  : "10000" // only single value accepted(highest value is 50000S)