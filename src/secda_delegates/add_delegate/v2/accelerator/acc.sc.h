#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include "add_pe.sc.h"
#include <systemc.h>

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<ADATA> din1;
  sc_fifo_out<ADATA> dout1;

  // GEMM 1 Inputs
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

  pe_container add_pe_array[2];


#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> computeS;
#else
  sc_signal<int> computeS;
#endif

  sc_out<int> computeSS;

  // Profiling variable
#ifndef __SYNTHESIS__
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles};
#endif

  // Functions
  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int);

  void send_parameters_ADD_PE(int, sc_fifo_in<ADATA> *);

  // HW Threads
  void Compute();

#ifndef __SYNTHESIS__
  void Counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) , add_pe_array() {
    
    // Connect ADD_PE ports
    for (int i = 0; i < 2; i++) {
      add_pe_array[i].init(clock, reset);
    }

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(Counter, clock);
    reset_signal_is(reset, true);
#endif

// #pragma HLS aggregate variable = add_pe_array

#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA1" port_map = {                                    \
      {din1_0 TDATA } {                                                        \
        din1_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA1" port_map = {{dout1_0 TDATA } {dout1_1 TLAST } }
  }
};

#endif