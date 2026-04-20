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

  int layer_t;

  // Index variables
  int i;
  int k;
  // int a;
  // int b;

  int pN_rem;
  int pM_rem;

  sc_fifo_in<ADATA> din1;
  sc_fifo_in<ADATA> din2;
  sc_fifo_in<ADATA> din3;
  sc_fifo_in<ADATA> din4;

  sc_fifo_out<ADATA> dout1;
  sc_fifo_out<ADATA> dout2;
  sc_fifo_out<ADATA> dout3;
  sc_fifo_out<ADATA> dout4;

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
  // int cur_crf;
  // int cur_crx;
  sc_int<32> crf_v[pmF];
  // sc_int<8> crx_v[pmF];
  sc_int<8> crx_v[pmF];
  int ra;
  int rhs_offset;
  int lhs_offset;

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  // TODO: This is rather hardcoded, the 64 is a max it could be and the
  // ... 1000 is just a stupid big number
  sc_int<8> rows[pnF][768]; // vit: 768,  deit:
  sc_int<8> cols[pmF][768];
  int temp[pkF][4];

  ADATA d_array[pmQ];
  // sc_int<8> cur_outs[4];
  sc_int<8> dout_1;
  sc_int<8> dout_2;
  sc_int<8> dout_3;
  sc_int<8> dout_4;
  int res[pnF][pmF];

  int bias[pmF];
  int wt_sum[pmF];
  int in_sum[pnF];
  int prec[pnF][pmF];

  int no_rows;
  int no_cols;

  int is_bias;

  // int layer = 0; // layer number

  ADATA d1;
  ADATA d2;
  ADATA d3;
  ADATA d4;

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

  int Quantised_Multiplier(int, sc_int<32>, sc_int<8>); // Defined in "compute.sc.h"

  sc_int<64> mul_s64(int, sc_int<64>);

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

// #pragma HLS array_partition variable = cur_outs complete dim = 0
#pragma HLS array_partition variable = res cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = prec cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = temp complete dim = 0
#pragma HLS array_partition variable = rows cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = cols cyclic dim = 1 factor = 4
#pragma HLS array_partition variable = cols cyclic dim = 2 factor = 32
#pragma HLS array_partition variable = d_array complete dim = 0
    // What is the best way to parition res?
    // #pragma HLS array_partition variable = res complete dim = 2

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
