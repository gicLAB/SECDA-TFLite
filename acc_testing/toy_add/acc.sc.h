#ifndef ACCNAME_H
#define ACCNAME_H

// #include "sysc_integrator/sysc_types.h"
#include <systemc.h>
#include "axi_support/axi_api_v2.h"
#include "sysc_profiler/profiler.h"

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

#define ACCNAME TOY_ADD
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define STOPPER -1

#define IN_BUF_LEN 4096
#define WE_BUF_LEN 8192
#define SUMS_BUF_LEN 1024

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648

#define MAX8 127
#define MIN8 -128

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> computeS;
#else
  sc_signal<int> computeS;
#endif

  // Profiling variable
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles};

  void Compute();

  void Counter();

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Counter, clock);
    reset_signal_is(reset, true);

    // clang-format off
#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA1" port_map = {                                    \
      {din1_0 TDATA } {                                                        \
        din1_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA1" port_map = {{dout1_0 TDATA } {dout1_1 TLAST } }
    // clang-format on
  }
};

#endif