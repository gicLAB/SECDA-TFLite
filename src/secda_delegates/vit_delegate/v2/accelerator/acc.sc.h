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

  sc_out<int> out_sig;

  // Index variables
  int i;
  int k;
  int a;
  int b;

  int pN_rem;
  int pM_rem;

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

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  sc_int<8> rows[64][512]; 
  sc_int<8> cols[64][512];

  DATA d_array[16]; // TODO: Size here is incorrect
  sc_int<8> cur_outs[4];

  int res[64][64];

  int no_rows;
  int no_cols;

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

  // int d1;
  // int d2;
  // int d3;
  // int d4;

#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> computeS; // ComputSignal
  sc_signal<int, SC_MANY_WRITERS> readS; // ReadSignal
  sc_signal<int, SC_MANY_WRITERS> PES; // Processing Element Signal
#else
  sc_signal<int> computeS;
  sc_signal<int> readS;
  sc_signal<int> PES;
#endif

#ifndef __SYNTHESIS__
  ClockCycles *per_batch_cycles = new ClockCycles("per_batch_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  ClockCycles *read_cycles = new ClockCycles("read_cycles", true);
  ClockCycles *PE_cycles = new ClockCycles("PE_cycles", true);
  std::vector<Metric *> profiling_vars = {per_batch_cycles, active_cycles,
   read_cycles, PE_cycles};
#endif

  void PE(int, int); // Defined in "PE.sc.h"

  sc_int<32> mul_s8(sc_int<8>, sc_int<8>); // Defined in "compute.sc.h"

  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int); // Defined in "compute.sc.h"

  int Quantised_Multiplier(int, int, int); // Defined in "compute.sc.h"

  void ReadRows();

  void ReadCols();

  void ReadBias();

  void PE_Post(int, int, int, int, int); // Defined in "pe_post.sc.h"


  // Hardware Threads
  void Compute(); // Defined in "compute.sc.h"

  // Counter for simulation
#ifndef __SYNTHESIS__
  void Counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

    // Cthreads are small circuits which have access to everything inside the 
    // ...sc_module but what is defined in the sc_cthread cannot be accessed 
    // .... by other cthreads

    SC_CTHREAD(Compute, clock);
    // reset_signal_is(reset, true);

    // SC_CTHREAD(ReadRows, clock);

    // SC_CTHREAD(ReadCols, clock);

    // SC_CTHREAD(ReadBias, clock)

    // SC_CTHREAD(PE, clock);

    // SC_CTHREAD(PE_Post, clock);

    // Counter for simulation
#ifndef __SYNTHESIS__
    SC_CTHREAD(Counter, clock);
    // reset_signal_is?
#endif

#pragma HLS array_partition variable = rows cyclic dim = 2 factor = 32 // pkF
#pragma HLS array_partition variable = cols cyclic dim = 1 factor = 4
#pragma HLS array_partition variable = cols cyclic dim = 2 factor = 32 // pkF
#pragma HLS array_partition variable = res cyclic dim = 2 factor = 4

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
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata =              \
  "-bus_bundle M_AXIS_DATA1" port_map = {                                    \
{ dout1_0 TDATA } {                                                             \
  dout1_1 TLAST } }
#pragma HLS RESOURCE variable = dout2 core = AXI4Stream metadata =             \
  "-bus_bundle M_AXIS_DATA2" port_map = {                                    \
{ dout2_0 TDATA } {                                                             \
  dout2_1 TLAST } }
#pragma HLS RESOURCE variable = dout3 core = AXI4Stream metadata = \
"-bus_bundle M_AXIS_DATA3" port_map = { \
  {dout3_0 TDATA} { \
    dout3_1 TLAST} } 
#pragma HLS RESOURCE variable = dout4 core = AXI4Stream metadata =             \
  "-bus_bundle M_AXIS_DATA4" port_map = { \
    {dout4_0 TDATA } { \
      dout4_1 TLAST} }
  }
};
#endif // ACCNAME_H
