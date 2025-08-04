# SECDA-TFLite Tutorials

This folder contains six tutorials that guide you through the process of using the SECDA-TFLite toolkit for various tasks such as simulation, hardware synthesis, and deployment on different boards. Each tutorial is designed to help you understand the capabilities of SECDA-TFLite and how to effectively use it in your projects.

## Tutorials Overview

1. **[Delegate Generation](1_delegate_generation/)** ([README](1_delegate_generation/README.md)) - Learn how to generate SECDA-TFLite delegates using the delegate generator tools. This tutorial covers creating custom delegates for your specific hardware accelerators.

2. **[Delegate Simulation](2_delegate_simulation/)** ([README](2_delegate_simulation/README.md)) - Explore how to simulate SECDA-TFLite delegates to validate their functionality before hardware implementation. This includes running simulations and debugging your delegates.

3. **[SECDA-TFLite Profiler](3_secda_tflite_profilier/)** ([README](3_secda_tflite_profilier/README.md)) - Understand how to use the SECDA-TFLite profiler to analyze performance characteristics of your delegates and accelerators, including both hardware and simulation profiling.

4. **[Hardware Automation](4_hardware_automation/)** ([README](4_hardware_automation/README.md)) - Learn how to use the hardware automation tools to generate hardware accelerator implementations from your delegate designs, including HLS generation and synthesis flows.

5. **[Benchmarking Suite](5_benchmarking_suite/)** ([README](5_benchmarking_suite/README.md)) - Discover how to use the benchmarking suite to evaluate the performance of your SECDA-TFLite delegates and accelerators across different models and configurations.

6. **[Full Design Process](6_full_design_process/)** ([README](6_full_design_process/README.md)) - A comprehensive tutorial that takes you through the complete end-to-end process of designing, implementing, and deploying a hardware accelerator using SECDA-TFLite.

## Getting Started

Each tutorial is self-contained and includes:
- Step-by-step instructions
- Example configurations
- Sample code and notebooks
- Expected outputs and results

It is recommended to follow the tutorials in order, as later tutorials may build upon concepts introduced in earlier ones.

## Prerequisites

Before starting these tutorials, ensure you have:
- Completed the SECDA-TFLite installation process
- Basic understanding of deep neural networks
- Familiarity with TensorFlow Lite
- Access to the required development tools (Vivado, Vitis, etc. as mentioned in the main README)