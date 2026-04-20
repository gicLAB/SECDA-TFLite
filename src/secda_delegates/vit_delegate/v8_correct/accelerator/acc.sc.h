#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include <systemc.h>
#include <vector>

#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif

SC_MODULE(ACCNAME) {

  sc_in<bool> clock;
  sc_in<bool> reset;

  // FIX: Must be sc_signal so Scheduler can write to it
  sc_signal<int> mode;

  sc_out<int> out_sig;

  sc_int<8> layer_t;

  // Index variables
  int i;
  int k;

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

  // Registers
  int pN;
  int pM;
  int pK;

  int crf;
  int crx;

  // Partitioned Params for 4 Cores
  sc_int<32> crf_v[NUM_CORES][pm_block];

  sc_int<8> crx_v[NUM_CORES][pm_block];

  int ra;
  int rhs_offset;
  int lhs_offset;

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  // --- MEMORY BUFFERS (Partitioned for Quad-Core) ---

  // Input Buffer: 4 Banks of [64 rows][1024 depth]
  // Dim 1 (Cores) is complete partitioned. Dim 3 (Data) is cyclic for SIMD.
  sc_int<8> rows[NUM_CORES][pn_block][max_pk];

  // Weight Buffer: 4 Banks of [64 cols][1024 depth]
  sc_int<8> cols[NUM_CORES][pm_block][max_pk];

  // Intermediates
  // Note: 'temp' is local to PE, usually better to declare inside PE loop
  // but if global, needs partitioning.
  // int temp[pkF][4];

  // ADATA d_array[pm_block / 4];

  sc_int<8> dout_1;
  sc_int<8> dout_2;
  sc_int<8> dout_3;
  sc_int<8> dout_4;

  // Result Buffer: 3D [Cores][Rows][Cols]
  int res[NUM_CORES][pn_block][pm_block];

  // Bias and Sums: Partitioned per Core
  int bias[NUM_CORES][pm_block];

  int wt_sum[NUM_CORES][pm_block];

  int in_sum[NUM_CORES][pn_block];

  // int prec[NUM_CORES][pn_block][pm_block];

  int no_rows;
  int no_cols;
  sc_int<8> is_bias;

  ADATA d1;
  ADATA d2;
  ADATA d3;
  ADATA d4;

  // int value1;
  // int value2;
  // int value3;
  // int value4;

  // sc_int<64> svalue1;
  // sc_int<64> svalue2;
  // sc_int<64> svalue3;
  // sc_int<64> svalue4;


  int temp_reg[NUM_CORES][pkF][4];

#ifndef __SYNTHESIS__
  // Profiling signals
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

  sc_signal<int, SC_MANY_WRITERS> inpReadReadyS;
  sc_signal<int, SC_MANY_WRITERS> wgtReadReadyS;
  sc_signal<int, SC_MANY_WRITERS> biasReadReadyS;
  sc_signal<int, SC_MANY_WRITERS> peReadyS;
  sc_signal<int, SC_MANY_WRITERS> ppuReadyS;

#else
  // Profiling signals
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

  sc_signal<int> inpReadReadyS;
  sc_signal<int> wgtReadReadyS;
  sc_signal<int> biasReadReadyS;
  sc_signal<int> peReadyS;
  sc_signal<int> ppuReadyS;
#endif

#ifndef __SYNTHESIS__
  ClockCycles *readTimeParam = new ClockCycles("readTimeParam", true);
  ClockCycles *readTimeInp = new ClockCycles("readTimeInp", true);
  ClockCycles *readTimeWgt = new ClockCycles("readTimeWgt", true);
  ClockCycles *readTimeBias = new ClockCycles("readTimeBias", true);
  ClockCycles *peTotal = new ClockCycles("peTotal", true);
  ClockCycles *pePostTotal = new ClockCycles("pePostTotal", true);

  std::vector<Metric *> profiling_vars = {
      readTimeParam, readTimeInp, readTimeWgt,
      readTimeBias,  peTotal,     pePostTotal,
  };
#endif

  // Functions
  void PE();
  sc_int<32> mul_s8(sc_int<8>, sc_int<8>);
  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int);

#ifndef __SYNTHESIS__
  int Quantised_Multiplier_Conv(int, sc_int<32>, sc_int<8>);
  int Quantised_Multiplier_FC(int x, int qm, int shift);
#else
  int Quantised_Multiplier_Conv(int, int, sc_int<64>, sc_int<32>, sc_int<32>,
                                sc_int<32>);
  int Quantised_Multiplier_FC(int x, int qm, int shift);
#endif

  sc_int<64> mul_s64(int, sc_int<64>);

  void ReadInp();
  void ReadWgt();
  void ReadBias();
  void PPU();
  void Scheduler();

#ifndef __SYNTHESIS__
  void Counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

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

#ifndef __SYNTHESIS__
    SC_CTHREAD(Counter, clock);
#endif

#pragma HLS array_partition variable = rows complete dim = 1
#pragma HLS array_partition variable = rows cyclic factor = 8 dim = 3

#pragma HLS array_partition variable = cols complete dim = 1
#pragma HLS array_partition variable = cols cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = cols cyclic factor = 8 dim = 3

// --- Result Buffers ---
#pragma HLS array_partition variable = res complete dim = 1
#pragma HLS array_partition variable = res cyclic factor = 4 dim = 3

// --- Small Control Arrays ---
#pragma HLS array_partition variable = in_sum complete dim = 1
// (in_sum only needs 1 read per cycle since it uses 'i', not 'j')

// FIX: Add cyclic factor 4 to dimension 2 so we can read j+0, j+1, j+2, j+3 at the same time
#pragma HLS array_partition variable = wt_sum complete dim = 1
#pragma HLS array_partition variable = wt_sum cyclic factor = 4 dim = 2

#pragma HLS array_partition variable = bias complete dim = 1
#pragma HLS array_partition variable = bias cyclic factor = 4 dim = 2

#pragma HLS array_partition variable = crf_v complete dim = 1
#pragma HLS array_partition variable = crf_v cyclic factor = 4 dim = 2

#pragma HLS array_partition variable = crx_v complete dim = 1
#pragma HLS array_partition variable = crx_v cyclic factor = 4 dim = 2



// (You can delete the old local_crf/crx/res partitions from here since we 
// moved them to be local variables inside the PPU.cpp function!)
#pragma HLS ARRAY_PARTITION variable = temp_reg complete dim = 0
#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA1" port_map = {{din1_0 TDATA } {din1_1 TLAST } }
#pragma HLS RESOURCE variable = din2 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA2" port_map = {{din2_0 TDATA } {din2_1 TLAST } }
#pragma HLS RESOURCE variable = din3 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA3" port_map = {{din3_0 TDATA } {din3_1 TLAST } }
#pragma HLS RESOURCE variable = din4 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA4" port_map = {{din4_0 TDATA } {din4_1 TLAST } }
#pragma HLS RESOURCE variable = dout1 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA1" port_map = {{dout1_0 TDATA } {dout1_1 TLAST } }
#pragma HLS RESOURCE variable = dout2 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA2" port_map = {{dout2_0 TDATA } {dout2_1 TLAST } }
#pragma HLS RESOURCE variable = dout3 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA3" port_map = {{dout3_0 TDATA } {dout3_1 TLAST } }
#pragma HLS RESOURCE variable = dout4 core = AXI4Stream metadata =             \
    "-bus_bundle M_AXIS_DATA4" port_map = {{dout4_0 TDATA } {dout4_1 TLAST } }
  }
};
#endif // ACCNAME_H