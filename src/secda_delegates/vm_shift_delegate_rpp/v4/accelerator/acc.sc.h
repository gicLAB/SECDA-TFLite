#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include <systemc.h>

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<DATA> din1;
  sc_fifo_in<DATA> din2;
  sc_fifo_in<DATA> din3;
  sc_fifo_in<DATA> din4;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

  unsigned int depth;
  unsigned int wgt_rows;
  unsigned int inp_rows;

#ifndef __SYNTHESIS__

  sc_signal<bool, SC_MANY_WRITERS> load_data;
  sc_signal<bool, SC_MANY_WRITERS> load_wgt;
  sc_signal<bool, SC_MANY_WRITERS> load_inp;

  sc_signal<bool, SC_MANY_WRITERS> schedule;
  sc_signal<bool, SC_MANY_WRITERS> out_check;
  sc_signal<int, SC_MANY_WRITERS> arr_check;

#else

  sc_signal<bool> load_data;
  sc_signal<bool> load_wgt;
  sc_signal<bool> load_inp;

  sc_signal<bool> schedule;
  sc_signal<bool> out_check;
  sc_signal<int> arr_check;
#endif

  // Global Inputs
  ACC_DTYPE<32> inp_1a_1[4096];
  ACC_DTYPE<32> inp_1b_1[4096];
  ACC_DTYPE<32> inp_1c_1[4096];
  ACC_DTYPE<32> inp_1d_1[4096];

  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data1a[4096];
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data2a[4096];
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data3a[4096];
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data4a[4096];

  ACC_DTYPE<32> out[16][4];
  ACC_DTYPE<32> g[16];

  ACC_DTYPE<32> test[3];

  sc_signal<int> w1S;
  sc_signal<int> w2S;
  sc_signal<int> w3S;
  sc_signal<int> w4S;

  sc_out<int> inS;
  sc_out<int> read_cycle_count;
  sc_out<int> process_cycle_count;
  sc_out<int> gemm_1_idle;
  sc_out<int> gemm_2_idle;
  sc_out<int> gemm_3_idle;
  sc_out<int> gemm_4_idle;
  sc_out<int> gemm_1_write;
  sc_out<int> gemm_2_write;
  sc_out<int> gemm_3_write;
  sc_out<int> gemm_4_write;
  sc_out<int> gemm_1;
  sc_out<int> gemm_2;
  sc_out<int> gemm_3;
  sc_out<int> gemm_4;
  sc_out<int> wstall_1;
  sc_out<int> wstall_2;
  sc_out<int> wstall_3;
  sc_out<int> wstall_4;

  sc_out<int> outS;
  sc_out<int> w1SS;
  sc_out<int> w2SS;
  sc_out<int> w3SS;
  sc_out<int> w4SS;
  sc_out<int> schS;
  sc_out<int> p1S;

  // struct var_array4 vars;

#ifndef __SYNTHESIS__
  // Profiling variable
  ClockCycles *cycles = new ClockCycles("cycles", true);
  ClockCycles *load_inps = new ClockCycles("load_inps", true);
  ClockCycles *load_wgts = new ClockCycles("load_wgts", true);
  ClockCycles *compute = new ClockCycles("compute", true);
  ClockCycles *idle1 = new ClockCycles("idle1", true);
  ClockCycles *idle2 = new ClockCycles("idle2", true);
  ClockCycles *idle3 = new ClockCycles("idle3", true);
  ClockCycles *idle4 = new ClockCycles("idle4", true);
  ClockCycles *post1 = new ClockCycles("post1", true);
  ClockCycles *post2 = new ClockCycles("post2", true);
  ClockCycles *post3 = new ClockCycles("post3", true);
  ClockCycles *post4 = new ClockCycles("post4", true);
  ClockCycles *gemm1 = new ClockCycles("gemm1", true);
  ClockCycles *gemm2 = new ClockCycles("gemm2", true);
  ClockCycles *gemm3 = new ClockCycles("gemm3", true);
  ClockCycles *gemm4 = new ClockCycles("gemm4", true);
  ClockCycles *wstall1 = new ClockCycles("wstall1", true);
  ClockCycles *wstall2 = new ClockCycles("wstall2", true);
  ClockCycles *wstall3 = new ClockCycles("wstall3", true);
  ClockCycles *wstall4 = new ClockCycles("wstall4", true);
  BufferSpace *gweightbuf_p = new BufferSpace("gweightbuf_p", GINP_BUF_LEN);
  BufferSpace *inputbuf_p = new BufferSpace("inputbuf_p", WGT_BUF_LEN);
  BufferSpace *weightbuf_p = new BufferSpace("weightbuf_p", INP_BUF_LEN);
  DataCountArray *gmacs = new DataCountArray("gmacs", 4);
  DataCountArray *gouts = new DataCountArray("gouts", 4);

  SignalTrack *shS = new SignalTrack("T_shS", true);
  SignalTrack *gmSA = new SignalTrack("T_gmSA", true);
  SignalTrack *gmSB = new SignalTrack("T_gmSB", true);
  SignalTrack *gmSC = new SignalTrack("T_gmSC", true);
  SignalTrack *gmSD = new SignalTrack("T_gmSD", true);
  SignalTrack *psSA = new SignalTrack("T_psSA", true);
  SignalTrack *psSB = new SignalTrack("T_psSB", true);
  SignalTrack *psSC = new SignalTrack("T_psSC", true);
  SignalTrack *psSD = new SignalTrack("T_psSD", true);

  std::vector<Metric *> profiling_vars = {shS, gmSB, psSB};

#endif

  void VM_PE(ACC_DTYPE<WGT_BRAM_DATAWIDTH> *, ACC_DTYPE<WGT_BRAM_DATAWIDTH> *,
             ACC_DTYPE<WGT_BRAM_DATAWIDTH> *, ACC_DTYPE<WGT_BRAM_DATAWIDTH> *,
             ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
             ACC_DTYPE<32>[][4], int, int, int);

#if defined(MUL88)
  sc_int<PROD_DATA_WIDTH> mul_s8(sc_int<8>, sc_int<8>);
#else
  sc_int<PROD_DATA_WIDTH> mul_s8(sc_int<4>, sc_int<8>);
#endif

  void Input_Handler();

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

    SC_CTHREAD(Input_Handler, clock);
    reset_signal_is(reset, true);

#pragma HLS array_partition variable = out complete dim = 0
// #pragma HLS array_partition variable = g complete dim = 0

#pragma HLS RESOURCE variable = din1 core = AXI4Stream metadata =              \
    "-bus_bundle S_AXIS_DATA1" port_map = {                                    \
      {din1_0 TDATA } {                                                        \
        din1_1 TLAST } }
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
#pragma HLS RESET variable = reset
  }
};
#endif /* ACCNAME_H */
