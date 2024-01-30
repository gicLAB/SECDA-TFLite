# SECDA Benchmarking Suite
The purpose of this suite is to quickly test for correctness of results and evaluate performance in terms of latency of SECDA-TFLite accelerators

## Currently support acceleratators

## Supported FPGAs

## Supported Models



## How to use the benchmarking suite


## Metrics produced



# Binary Generation
This contains a very simple script used to generate and sent binaries for the supported delegates + tool combo
- compile.ipynb is currently used to create the generate.sh which is actually going to be doing binary generations
- editting compile.ipynb to supported delegates and tools dictionary is the current best way to keeping version control

This can be used in combination with secda_benchmarking_suite, benchmark new (and older) delegates/accelerator

