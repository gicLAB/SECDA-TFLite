#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include "hw_submodule.sc.h"
#include <systemc.h>

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  // ================================================= //
  // Global ports
  // ================================================= //

  // Data ports
  sc_fifo_in<ADATA> din1;
  sc_fifo_out<ADATA> dout1;
  sc_out<int> computeSS;

  // ================================================= //
  // Global variables
  // ================================================= //
  int lshift;
  int in1_off;
  int in1_sv;
  int in1_mul;
  int in2_off;
  int in2_sv;
  int in2_mul;
  int out1_off;
  int out1_sv;
  int out1_mul;
  int qa_max;
  int qa_min;

  // ================================================= //
  // Global buffers
  // ================================================= //

  // ================================================= //
  // Global signals
  // ================================================= //
  DEFINE_SC_SIGNAL(int, computeS);

  // ================================================= //
  // Functions
  // ================================================= //

  // Functions
  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int);

  void send_parameters_tempdel_PE(int, sc_fifo_in<ADATA> *);

  // ================================================= //
  // HW Threads
  // ================================================= //

  void Compute();

#ifndef __SYNTHESIS__
  void Simulation_Profiler();
#endif

  // ================================================= //
  // Submodules
  // ================================================= //

  struct hw_submodule_var_array hw_submodule_array;

  // ================================================= //
  // Profiling variable
  // ================================================= //

#ifndef __SYNTHESIS__
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles};
#endif
  // ================================================= //

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

    // Connect tempdel_PE ports
    hw_submodule_array.init(clock, reset);

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(Simulation_Profiler, clock);
    reset_signal_is(reset, true);
#endif

    SLV_Prag(computeSS);
    AXI4S_In_Prag(din1);
    AXI4S_Out_Prag(dout1);
  }
};

#endif