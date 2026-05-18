# Connection to Ramulator2
## Status
- Currently builds base.h


# Changes made to support the build
- Added bazel BUILD files to Ramulator source folders
- Created ramulator_connector library which connect everything into on libs
- I had to remove the self referencing folder from includes within header and sources files ```#include 'dram/dram.h'```  => ```#include 'dram.h'```
- Updated C++ standard used throughout TFLite + SystemC build to C++20
  - First I changed .bazelrc line 315 to  ``` build:linux --cxxopt=-std=c++20```
  - Also updated systemc.BUILD files  to C++20 (two places are changed)
  - Also removed ```-fno-exceptions``` from tensorflow/lite/build_def.bzl's ```tflite_copts```
- Installed spdlog [https://github.com/gabime/spdlog] and yaml-cpp [https://github.com/jbeder/yaml-cpp]
    - Download, CMake and ```sudo make install```
  
  