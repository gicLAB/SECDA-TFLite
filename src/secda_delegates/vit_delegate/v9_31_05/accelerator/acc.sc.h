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

  sc_signal<int> mode;
  sc_out<int> out_sig;
  sc_int<8> layer_t;

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

  int pN;
  int pM;
  int pK;
  int crf;
  int crx;

  sc_int<32> crf_v[NUM_CORES][pm_block];
  sc_int<8> crx_v[NUM_CORES][pm_block];

  int ra;
  int rhs_offset;
  int lhs_offset;

  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  // OPTIMIZATION: Packed 32-bit arrays. Depth divided by 4.
  sc_int<32> rows_packed[NUM_CORES][pn_block][max_pk / 4];
  sc_int<32> cols_packed[NUM_CORES][pm_block][max_pk / 4];

  sc_int<8> dout_1;
  sc_int<8> dout_2;
  sc_int<8> dout_3;
  sc_int<8> dout_4;

  int res[NUM_CORES][pn_block][pm_block];
  int bias[NUM_CORES][pm_block];
  int wt_sum[NUM_CORES][pm_block];
  int in_sum[NUM_CORES][pn_block];

  int no_rows;
  int no_cols;
  sc_int<8> is_bias;

  ADATA d1;
  ADATA d2;
  ADATA d3;
  ADATA d4;

  int temp_reg[NUM_CORES][pkF][4];

#ifndef __SYNTHESIS__
  sc_signal<int, SC_MANY_WRITERS> accTotalS;
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
  sc_signal<int> accTotalS;
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
  ClockCycles *accTotal = new ClockCycles("accTotal", true);
  ClockCycles *readTimeParam = new ClockCycles("readTimeParam", true);
  ClockCycles *readTimeInp = new ClockCycles("readTimeInp", true);
  ClockCycles *readTimeWgt = new ClockCycles("readTimeWgt", true);
  ClockCycles *readTimeBias = new ClockCycles("readTimeBias", true);
  ClockCycles *peTotal = new ClockCycles("peTotal", true);
  ClockCycles *pePostTotal = new ClockCycles("pePostTotal", true);

  std::vector<Metric *> profiling_vars = {
      accTotal, readTimeParam, readTimeInp, readTimeWgt,
      readTimeBias,  peTotal,     pePostTotal,
  };
#endif

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

// OPTIMIZATION: Cyclic factor reduced to 4 to save BRAMs.
#pragma HLS array_partition variable = rows_packed complete dim = 1
#pragma HLS array_partition variable = rows_packed cyclic factor = 4 dim = 3

#pragma HLS array_partition variable = cols_packed complete dim = 1
#pragma HLS array_partition variable = cols_packed cyclic factor = 4 dim = 2
#pragma HLS array_partition variable = cols_packed cyclic factor = 4 dim = 3

#pragma HLS array_partition variable = res complete dim = 1
#pragma HLS array_partition variable = res cyclic factor = 4 dim = 3

#pragma HLS array_partition variable = in_sum complete dim = 1

#pragma HLS array_partition variable = wt_sum complete dim = 1
#pragma HLS array_partition variable = wt_sum cyclic factor = 4 dim = 2

#pragma HLS array_partition variable = bias complete dim = 1
#pragma HLS array_partition variable = bias cyclic factor = 4 dim = 2

#pragma HLS array_partition variable = crf_v complete dim = 1
#pragma HLS array_partition variable = crf_v cyclic factor = 4 dim = 2

#pragma HLS array_partition variable = crx_v complete dim = 1
#pragma HLS array_partition variable = crx_v cyclic factor = 4 dim = 2

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