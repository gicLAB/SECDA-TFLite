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

  // sc_in<int> in_sig;
  sc_out<int> out_sig;

  // Index variables
  int i;
  int k;
  // int a;
  // int b;

  int pN_rem;
  int pM_rem;
  int pK_rem;

  sc_fifo_in<DATA> din1;
  sc_fifo_in<DATA> din2;
  sc_fifo_in<DATA> din3;
  sc_fifo_in<DATA> din4;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

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

  int is_first_k;
  int is_last_k;

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  sc_int<8> rows[pn_tile][pk_tile]; 
  sc_int<8> cols[pm_tile][pk_tile];
  int temp[pe_tile][4];

  DATA d_array[pm_tile / 4];
  // sc_int<8> cur_outs[4];
  sc_int<8> dout_1;
  sc_int<8> dout_2;
  sc_int<8> dout_3;
  sc_int<8> dout_4;
  int res[pn_tile][pm_tile];

  int bias[pm_tile];
  int wt_sum[pm_tile];
  int in_sum[pn_tile];
  int prec[pn_tile][pm_tile];

  int no_rows;
  int no_cols;

  int is_bias;

  // int layer = 0; // layer number

  DATA d1;
  DATA d2;
  DATA d3;
  DATA d4;

  int value1;
  int value2;
  int value3;
  int value4;

  sc_int<64> svalue1;
  sc_int<64> svalue2;
  sc_int<64> svalue3;
  sc_int<64> svalue4;

  // Profiling vars
  int no_inpR = 0;  // number of input reads
  int no_wgtR = 0;  // number of weight reads
  int no_biasR = 0; // number of bias reads
  int no_pe = 0;    // number of PEs done (same as ppu)

#ifndef __SYNTHESIS__
  // Profiling signals
  // sc_signal<int, SC_MANY_WRITERS> readTimeTotalS;
  sc_signal<int, SC_MANY_WRITERS> readTimeParamS;
  sc_signal<int, SC_MANY_WRITERS> readTimeInpS;
  sc_signal<int, SC_MANY_WRITERS> readTimeWgtS;
  sc_signal<int, SC_MANY_WRITERS> readTimeBiasS;

  sc_signal<int, SC_MANY_WRITERS> peTotalS;
  sc_signal<int, SC_MANY_WRITERS> peMultiplyS;
  sc_signal<int, SC_MANY_WRITERS> peAccumulateS;

  sc_signal<int, SC_MANY_WRITERS> pePostTotalS;
  sc_signal<int, SC_MANY_WRITERS> pePostProcessS;
  sc_signal<int, SC_MANY_WRITERS> pePostWriteS;

  // Implementation signals
  sc_signal<int, SC_MANY_WRITERS> inpReadReadyS;

  sc_signal<int, SC_MANY_WRITERS> wgtReadReadyS;

  sc_signal<int, SC_MANY_WRITERS> biasReadReadyS;

  sc_signal<int, SC_MANY_WRITERS> peReadyS;

  sc_signal<int, SC_MANY_WRITERS> ppuReadyS;

  sc_signal<int, SC_MANY_WRITERS> is_first_kS;

#else
  // Profiling signals
  // sc_signal<int> readTimeTotalS;
  sc_signal<int> readTimeParamS;
  sc_signal<int> readTimeInpS;
  sc_signal<int> readTimeWgtS;
  sc_signal<int> readTimeBiasS;

  sc_signal<int> peTotalS;
  sc_signal<int> peMultiplyS;
  sc_signal<int> peAccumulateS;

  sc_signal<int> pePostTotalS;
  sc_signal<int> pePostProcessS;
  sc_signal<int> pePostWriteS;

  // Implementation signals
  sc_signal<int> inpReadReadyS;

  sc_signal<int> wgtReadReadyS;

  sc_signal<int> biasReadReadyS;

  sc_signal<int> peReadyS;

  sc_signal<int> ppuReadyS;

  sc_signal<int> is_first_kS;
#endif

#ifndef __SYNTHESIS__
  // ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  // ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  // ClockCycles *read_cycles = new ClockCycles("read_cycles", true);
  // ClockCycles *PE_cycles = new ClockCycles("PE_cycles", true);
  // std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles,
  //  read_cycles, PE_cycles};
  // ClockCycles *readTimeTotal = new ClockCycles("readTimeTotal", true);
  ClockCycles *readTimeParam = new ClockCycles("readTimeParam", true);
  ClockCycles *readTimeInp = new ClockCycles("readTimeInp", true);
  ClockCycles *readTimeWgt = new ClockCycles("readTimeWgt", true);
  ClockCycles *readTimeBias = new ClockCycles("readTimeBias", true);
  ClockCycles *peTotal = new ClockCycles("peTotal", true);
  // ClockCycles *peMultiply = new ClockCycles("peMultiply", true);
  // ClockCycles *peAccumulate = new ClockCycles("peAccumulate", true);
  ClockCycles *pePostTotal = new ClockCycles("pePostTotal", true);
  // ClockCycles *pePostProcess = new ClockCycles("pePostProcess", true);
  // ClockCycles *pePostWrite = new ClockCycles("pePostWrite", true);
  std::vector<Metric *> profiling_vars = {
      // readTimeTotal,
      readTimeParam, readTimeInp, readTimeWgt, readTimeBias, peTotal,
      // peMultiply,
      // peAccumulate,
      pePostTotal,
      // pePostProcess,
      // pePostWrite
  };

#endif

  void PE(); // Defined in "PE.sc.h"

  sc_int<32> mul_s8(sc_int<8>, sc_int<8>); // Defined in "compute.sc.h"

  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int,
                              int); // Defined in "compute.sc.h"

  int Quantised_Multiplier(int, int, int); // Defined in "compute.sc.h"

  void ReadInp();

  void ReadWgt();

  void ReadBias();

  void PPU(); // Defined in "pe_post.sc.h"

  // Hardware Threads
  void Scheduler(); // Defined in "compute.sc.h"

  // Counter for simulation
#ifndef __SYNTHESIS__
  void Counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

    // Cthreads are small circuits which have access to everything inside the
    // ...sc_module but what is defined in the sc_cthread cannot be accessed
    // .... by other cthreads

    // vars.init(clock, reset);

    SC_CTHREAD(Scheduler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(ReadInp, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(ReadWgt, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(ReadBias, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(PE, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(PPU, clock);
    reset_signal_is(reset, true);

    // Counter for simulation
#ifndef __SYNTHESIS__
    SC_CTHREAD(Counter, clock);
    // reset_signal_is?
#endif

// #pragma HLS array_partition variable = res cyclic factor = 4 dim = 2
// #pragma HLS array_partition variable = prec cyclic factor = 4 dim = 2
// #pragma HLS array_partition variable = temp complete dim = 0
// #pragma HLS array_partition variable = rows cyclic dim = 2 factor = 32
// #pragma HLS array_partition variable = cols cyclic dim = 1 factor = 4
// #pragma HLS array_partition variable = cols cyclic dim = 2 factor = 32
// #pragma HLS array_partition variable = d_array complete dim = 0
//     // What is the best way to parition res?
// Place these pragmas inside your SC_MODULE definition in your accelerator's .h file

// --- Input and Weight Buffers ---
// These are the largest arrays. We partition them to feed the PE pipeline efficiently.

// Partition 'rows' cyclically along the second dimension (dim 2).
// The factor 'pe_tile' matches the pipeline depth in the PE, ensuring that
// the pipeline never stalls waiting for input data from the K dimension.
// PE TILE
#pragma HLS array_partition variable=rows cyclic factor=64 dim=2

// Partition 'cols' cyclically along the first dimension (dim 1).
// The factor '4' matches the 4-wide computation in the PE (j, j+1, j+2, j+3).
// This creates 4 smaller BRAMs, allowing simultaneous reads from four different columns.
#pragma HLS array_partition variable=cols cyclic factor=4 dim=1


// Partition 'res' cyclically along the second dimension (dim 2).
// This allows the PE and PPU to read/write 4 adjacent elements (j to j+3) in a single clock cycle.
#pragma HLS array_partition variable=res cyclic factor=4 dim=2

// Partition 'prec' in the same way as 'res', as it's read by the PPU
// in the same 4-wide pattern.
#pragma HLS array_partition variable=prec cyclic factor=4 dim=2

// --- Temporary and 1D Buffers ---
// These are small but are accessed frequently in pipelined loops.
// We partition them completely into registers to eliminate any memory bottlenecks.

// Completely partition the small 'temp' array in the PE. This is essential
// because the loops using it are unrolled, requiring full parallel access.
#pragma HLS array_partition variable=temp complete dim=0

// Completely partition all 1D arrays. This ensures that the pipelined loops
// in ReadBias and the PPU can access any element without delay.
#pragma HLS array_partition variable=bias complete dim=1
#pragma HLS array_partition variable=wt_sum complete dim=1
#pragma HLS array_partition variable=in_sum complete dim=1
#pragma HLS array_partition variable=d_array complete dim=1

#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA1" port_map = {                                    \
{ din1_0 TDATA } {                                                             \
  din1_1 TLAST } }
#pragma HLS RESOURCE variable = din2 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA2" port_map = {                                    \
{ din2_0 TDATA } {                                                             \
  din2_1 TLAST } }
#pragma HLS RESOURCE variable = din3 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA3" port_map = {                                    \
{ din3_0 TDATA } {                                                             \
  din3_1 TLAST } }
#pragma HLS RESOURCE variable = din4 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA4" port_map = {                                    \
{ din4_0 TDATA } {                                                             \
  din4_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA1" port_map = {                                    \
{ dout1_0 TDATA } {                                                            \
  dout1_1 TLAST } }
#pragma HLS RESOURCE variable = dout2 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA2" port_map = {                                    \
{ dout2_0 TDATA } {                                                            \
  dout2_1 TLAST } }
#pragma HLS RESOURCE variable = dout3 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA3" port_map = {                                    \
{ dout3_0 TDATA } {                                                            \
  dout3_1 TLAST } }
#pragma HLS RESOURCE variable = dout4 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA4" port_map = {                                    \
{ dout4_0 TDATA } {                                                            \
  dout4_1 TLAST } }
  }
};
#endif // ACCNAME_H
