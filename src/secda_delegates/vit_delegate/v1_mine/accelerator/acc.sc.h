#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include <systemc.h>
// #include
// "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
// #include
// "tensorflow/lite/delegates/utils/secda_tflite/secda_integrator/sysc_types.h"
// #include
// "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
// #include
// "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include <vector>

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

SC_MODULE(ACCNAME) {

  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<DATA> din;
  sc_fifo_out<DATA> dout;
  sc_out<int> computeSS;

  // Inputs here - like all of the calculation variables
  // TODO: Not sure if this is all I need here
  int N;
  int K;
  int M;

  int pN;
  int pM;
  int pK;

  int crf;
  int crx;
  int ra;
  int rhs_offset;
  int lhs_offset;

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  // TODO: This is rather hardcoded, the 64 is a max it could be and the
  // ... 1000 is just a stupid big number
  sc_int<8> rows[64][1000]; // 1000 Instead of pk
  sc_int<8> cols[64][1000];

  DATA d_array[16];
  sc_int<8> cur_outs[4];
  int res[64][64];

  int no_rows;
  int no_cols;

  // Data d;

#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> computeS;
  sc_signal<int, SC_MANY_WRITERS> readS;
#else
  sc_signal<int> computeS;
  sc_signal<int> readS;
#endif

#ifndef __SYNTHESIS__
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  ClockCycles *read_cycles = new ClockCycles("read_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles, read_cycles};
#endif

  // Functions

  sc_int<32> mul_s8(sc_int<8>, sc_int<8>);
  void PE(int, int);
  // ACC_DTYPE<32> Clamp_Combine(int, int, int, int);
  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int);
  int Quantised_Multiplier(int, int, int);

  // Hardware Threads
  void Compute(); // Defined in "compute.sc.h"

  // Counter for simulation
#ifndef __SYNTHESIS__
  void Counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

    // Cthreads

    SC_CTHREAD(Compute, clock);

    
    #ifndef __SYNTHESIS__
        SC_CTHREAD(Counter, clock);
        // reset_signal_is?
    #endif

#pragma HLS array_partition variable = res cyclic factor = 4 dim = 2

// #pragma HLS array_partition variable = rows cyclic dim = 1 factor = 16
#pragma HLS array_partition variable = cols cyclic dim = 1 factor = 4

#pragma HLS array_partition variable = rows cyclic dim = 2 factor = 32 // pkF
#pragma HLS array_partition variable = cols cyclic dim = 2 factor = 32 // pkF

#pragma HLS RESOURCE variable = din core = AXI4Stream metadata =               \
    "-bus_bundle S_AXIS_DATA1" port_map = {                                    \
{ din_0 TDATA } {                                                              \
  din_1 TLAST } }
#pragma HLS RESOURCE variable = dout core = AXI4Stream metadata =              \
    "-bus_bundle M_AXIS_DATA1" port_map = {                                    \
{ dout_0 TDATA } {                                                             \
  dout_1 TLAST } }
  }
};
#endif
