# Toy Add Accelator (WIP)
The ToyAdd accelerator is a simple accelerator which takes two signed 8-bit input matrices and adds them together, applies the re-quantization function and returns the resultant matrix.


# ToyAdd - Accelerator
The synthesizable source code for the ToyAdd accelerator can be found within the [secda_tfilte_accelerator](https://github.com/gicLAB/SECDA-TFLite/tree/main/secda_tflite_accelerator) folder.

The simple stream-based accelerator initially expects the following to perform the Add operation:
* Requantization metadata header
* Length of input matrices (both matrices must be of equal size)
* Matrix A and Matrix B interleaved between 4 elements


# ToyAdd - Simulation Delegate

The ToyAdd simulation delegate can be found within the [sim_delegate](https://github.com/gicLAB/SECDA-TFLite/tree/main/tutorial/toy_accelerator/sim_delegate) folder.  The simulation delegate contains the bare essential code to define a TFLite delegate along with the [driver](https://github.com/gicLAB/SECDA-TFLite/tree/main/tutorial/toy_accelerator/sim_delegate/accelerator/driver) code and the [SystemC accelerator definition](https://github.com/gicLAB/SECDA-TFLite/tree/main/tutorial/toy_accelerator/sim_delegate/accelerator) used for within the simulation. 

The simulation delegate utilizes SECDA-TFLite's [AXI API](https://github.com/gicLAB/SECDA-TFLite/tree/main/secda_tflite/axi_support) along with the SystemC [Integrator](https://github.com/gicLAB/SECDA-TFLite/tree/main/secda_tflite/sysc_integrator) and [Profiler](https://github.com/gicLAB/SECDA-TFLite/tree/main/secda_tflite/sysc_profiler) libraries to ease the development and integration of the accelerator within TFLite. Please follow the 1st part of the [step-by-step tutorial](part-1---simulation) for a more in-depth look into the simulation delegate.

# ToyAdd - FPGA Delegate

The ToyAdd FPGA delegate can be found within the [sim_delegate](https://github.com/gicLAB/SECDA-TFLite/tree/main/tutorial/toy_accelerator/sim_delegate) folder. The FPGA delegate contains the bare essential code to define a TFLite delegate along with the [driver](https://github.com/gicLAB/SECDA-TFLite/tree/main/tutorial/toy_accelerator/sim_delegate/accelerator/driver) code to offload onto an actual FPGA mapped accelerator.

The FPGA delegate utilizes SECDA-TFLite's [AXI API](https://github.com/gicLAB/SECDA-TFLite/tree/main/secda_tflite/axi_support) to ease the development and integration of the accelerator within TFLite.

Please follow the 2nd part of the [step-by-step tutorial](#part-2---fpga) for a more in-depth look into the FPGA delegate and running the accelerator of your PYNQ-Z1.


# ToyAdd - Step-by-Step Tutorial
This tutorial demonstrates how to develop a simple accelerator along with the simulation and FPGA delegates for TFLite using the SECDA-TFLite toolkit.
Before reading through this tutorial we highly recommend following (Docker container setup)[https://github.com/gicLAB/SECDA-TFLite] to have access and run the simulation code via VSCode.

## Part 1 - Simulation
The simulation delegate implements the simple delegate interface defined within TFLite and we would recommend any custom delegate development using SECDA-TFLite also uses the [simple delegate inteface](https://www.tensorflow.org/lite/performance/implementing_delegate). There is some general boilerplate code which is required to connect the delegate to TFLite but we will focus on ["toy_delegate.cc"](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc) which contains the core functionalities of the ToyAdd accelerator simulation delegate. The following steps discuss different aspects of the delegate code which can be adapted when defining your own simulation delegate.

### Step 1.1:
Declare some vital global variables:
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L20-L24

* [*del_param*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/secda_tflite/threading_utils/acc_helpers.h#L28) is used to define general metadata about the delegate. 
* *ACCNAME* is macro which is associated with the accelerator we are working with, this is defined in the [accelerator header](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/sim_delegate/accelerator/acc.sc.h). Creating a simple macro like this helps version control the different variation of the same accelerator where we change some aspects of the accelerator but the delegate and driver code does not require change.
* [*stream_dma*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/secda_tflite/axi_support/axi_api_v2.h#L120) is a DMA helper struct which allows the developer to utilize a single DMA engine to perform data transfers from and to the accelerator. In this case, we are utilizing the SystemC version of this struct which helps connect the accelerator to a [SystemC-based DMA engine](https://github.com/gicLAB/SECDA-TFLite/blob/main/secda_tflite/sysc_integrator/axi4s_engine.sc.h) defined within SECDA-TFLite.
* [*sysC_sigs*](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/sim_delegate/accelerator/driver/systemc_binding.h) is a custom developer-defined struct which varies depending on the accelerator definition. The purpose of this struct is to define the signals and ports utilized for this accelerator and to bind accelerator AXIS DMA ports to the streams_dma's DMA engine. The developer can use this definition as the template to develop their own sysC_sigs for their accelerator.
* [*Profile*](https://github.com/gicLAB/SECDA-TFLite/blob/main/secda_tflite/sysc_profiler/profiler.h) is the SystemC profiler which the developer can use to track different aspects of the delegate, driver and accelerator execution.

### Step 1.2:
Define the [*Init*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L36) functionality. The init function runs once per delegate partition (which could be multiple nodes or single nodes). To understand the delegate partition refer to the [TFLite docs](https://www.tensorflow.org/lite/performance/implementing_delegate). The first time we enter the *Init* function during runtime we initialize the SystemC modules (DMA, Accelerator), the simulation profiler and bind AXI ports from the DMA to the Accelerator, this binding process requires the developer to create the [systemC_binder](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L45) function.
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L38-L52
The remainder of the *Init* function depends on the type of node we want to process. In our case for each delegated node, we save the tensor index to both input tensors and the single output tensor along with some additional metadata required to process the node. We save this data to the delegate so that the delegate can access it during the remainder of the runtime.


### Step 1.3:
Define the [*Prepare*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L91) functionality. The prepare function runs once per delegated node before inference to prepare each node for execution. The prepare function is dependent on the type of node we want to process but typically creating a similar prepare function to the default TFLite implementation is recommended with some modification to prepare/reshape weight data preemptively before inference if the accelerator uses a specific data format.


### Step 1.4:
Define the [*Eval*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L153) functionality. The eval function runs once per delegated node during inference to execute the node. This is where we offload the node to be executed on our custom hardware accelerator. Before we call the accelerator driver we first initialize and pack the accelerator container with all the details required for the driver to offload the computation to the accelerator. Finally, we call the invoke the driver and pass the accelerator container ())
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L188-L208


### Step 1.5:
Define the [*IsNodeSupportedByDelegate*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/toy_delegate.cc#L246) functionality. The developer needs specific the TFLite node type and attributes the delegate supports. Before inference, TFLite checks every node within the target DNN model against this function to identify all the nodes that can be partitioned to this delegate.

### Step 1.6:
Define the [*Accelerator Container*](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/sim_delegate/accelerator/driver/acc_container.h), which is a custom developer-defined struct which varies depending on the accelerator definition. The purpose of the accelerator container is to capture all the metadata and TFLite tensor pointers for the current node being executed into a container before passing it on to the accelerator driver. For the ToyADD accelerator, we store the essential hardware and profiler pointers along with ADD node input and output tensor pointers along with some extra metadata.

### Step 1.7:
Define the [*Accelerator Driver*](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/sim_delegate/accelerator/driver/add_driver.h).
The ToyAdd driver first loads re-quantization metadata into the DMA input buffer and then loads the input buffer with the two input matrices.
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/accelerator/driver/add_driver.h#L10-L40
Once the data is prepared to be sent to the accelerator, the driver requests the DMA via the stream DMA interface to send the data and then waits for the accelerator to finish computation and return the output data.
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/accelerator/driver/add_driver.h#L42-L45
Finally, the driver saves the profiling info from the accelerator simulation and unloads the resultant matrix from the DMA output buffer to the appropriate TFLite output tensor.
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/sim_delegate/accelerator/driver/add_driver.h#L47-L52


## Part 2 - FPGA
The FPGA delegate implements the simple delegate interface defined within TFLite and we would recommend any custom delegate development using SECDA-TFLite also uses the [simple delegate inteface](https://www.tensorflow.org/lite/performance/implementing_delegate). There is some general boilerplate code which is required to connect the delegate to TFLite but we will focus on ["toy_delegate.cc"](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc) which contains the core functionalities of the ToyAdd accelerator simulation delegate. The following steps discuss different aspects of the delegate codes which can be adapted when defining your own FPGA delegate.
Note: Highly recommend reading part 1 since it will explain details which are common to both parts.

### Step 2.1:
Define some vital global variables:
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L15-L18
Similar to the simulation delegate we declare the *del_params* and *stream_dma* structs, additional we declare also the *toy_times* struct which is a simple container for tracking runtime measurements of time in milliseconds.

### Step 2.2:
Define the [*Init*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L30)
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L33-L47
The first time we enter the *Init* function during runtime we initialize the stream_dma with the address of the accelerator, input buffers, output buffers and the size of the buffers. We also [mmap the accelerator control/status register space](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L34) in which can be used later to send control signals to the accelerator or to read status signals from it. 
The remainder of the *Init* function is identical to the simulation version from part 1.

### Step 2.3:
Define the [*Prepare*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L87). The *Prepare* function is identical to the simulation version from part 1. 


### Step 2.4:
Define the [*Eval*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L149).
The *Eval* function is almost identical to the simulation version from part 1, except for prf_start and prf_end calls which are preprocessor macros defined to easily benchmark the duration of different parts of code. 
https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L203-L205
Notice we save the elapsed time into "toy_t.driver_time" which later is printed out once all delegates are computed in [line 210](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L210)


### Step 2.5:
Define the [*IsNodeSupportedByDelegate*](https://github.com/gicLAB/SECDA-TFLite/blob/153d388ec6af6de85594cb9bb96900d8c16417e5/tutorial/toy_accelerator/fpga_delegate/toy_delegate.cc#L236).
The *IsNodeSupportedByDelegate* function is identical to the simulation version from part 1. 


### Step 2.6:
Define the [*Accelerator Container*](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/fpga_delegate/driver/acc_container.h)
The *Accelerator Container* is almost identical to the simulation version from part 1, except that in the FPGA version there is no need for pointers to the accelerator or profiler.

### Step 2.7:
Define the [*Accelerator Driver*](https://github.com/gicLAB/SECDA-TFLite/blob/main/tutorial/toy_accelerator/sim_delegate/accelerator/driver/add_driver.h).
The *Accelerator Driver* is almost identical to the simulation version from part 1, except that in the FPGA version we define the memory address associated with the accelerator and DMA input and output buffers along with the buffer size required per buffer in bytes.
