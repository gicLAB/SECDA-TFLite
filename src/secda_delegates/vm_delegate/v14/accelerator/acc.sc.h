#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include "vmm_unit.sc.h"
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
  unsigned int wgt_block;
  unsigned int inp_block;
  unsigned int unit_counter;

#ifndef __SYNTHESIS__

  sc_signal<bool, SC_MANY_WRITERS> load_data;
  sc_signal<bool, SC_MANY_WRITERS> load_wgt_drv;
  sc_signal<bool, SC_MANY_WRITERS> load_inp_drv;

  sc_signal<bool, SC_MANY_WRITERS> schedule;
  sc_signal<bool, SC_MANY_WRITERS> out_check;
  sc_signal<int, SC_MANY_WRITERS> arr_check;

#else

  sc_signal<bool> load_data;
  sc_signal<bool> load_wgt_drv;
  sc_signal<bool> load_inp_drv;

  sc_signal<bool> schedule;
  sc_signal<bool> out_check;
  sc_signal<int> arr_check;
#endif
  sc_fifo<int> arranger_fifo;

  // Global Inputs
  ACC_DTYPE<32> inp_data1[GINP_BUF_LEN];
  ACC_DTYPE<32> inp_data2[GINP_BUF_LEN];
  ACC_DTYPE<32> inp_data3[GINP_BUF_LEN];
  ACC_DTYPE<32> inp_data4[GINP_BUF_LEN];

  int ra;

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
  sc_out<int> gemm_5_idle;
  sc_out<int> gemm_6_idle;
  sc_out<int> gemm_7_idle;
  sc_out<int> gemm_8_idle;
  sc_out<int> gemm_9_idle;
  sc_out<int> gemm_10_idle;
  sc_out<int> gemm_11_idle;
  sc_out<int> gemm_12_idle;
  sc_out<int> gemm_13_idle;
  sc_out<int> gemm_14_idle;
  sc_out<int> gemm_15_idle;
  sc_out<int> gemm_16_idle;

  sc_out<int> gemm_1_write;
  sc_out<int> gemm_2_write;
  sc_out<int> gemm_3_write;
  sc_out<int> gemm_4_write;
  sc_out<int> gemm_5_write;
  sc_out<int> gemm_6_write;
  sc_out<int> gemm_7_write;
  sc_out<int> gemm_8_write;
  sc_out<int> gemm_9_write;
  sc_out<int> gemm_10_write;
  sc_out<int> gemm_11_write;
  sc_out<int> gemm_12_write;
  sc_out<int> gemm_13_write;
  sc_out<int> gemm_14_write;
  sc_out<int> gemm_15_write;
  sc_out<int> gemm_16_write;

  sc_out<int> gemm_1;
  sc_out<int> gemm_2;
  sc_out<int> gemm_3;
  sc_out<int> gemm_4;
  sc_out<int> gemm_5;
  sc_out<int> gemm_6;
  sc_out<int> gemm_7;
  sc_out<int> gemm_8;
  sc_out<int> gemm_9;
  sc_out<int> gemm_10;
  sc_out<int> gemm_11;
  sc_out<int> gemm_12;
  sc_out<int> gemm_13;
  sc_out<int> gemm_14;
  sc_out<int> gemm_15;
  sc_out<int> gemm_16;

  sc_out<int> wstall_1;
  sc_out<int> wstall_2;
  sc_out<int> wstall_3;
  sc_out<int> wstall_4;
  sc_out<int> wstall_5;
  sc_out<int> wstall_6;
  sc_out<int> wstall_7;
  sc_out<int> wstall_8;
  sc_out<int> wstall_9;
  sc_out<int> wstall_10;
  sc_out<int> wstall_11;
  sc_out<int> wstall_12;
  sc_out<int> wstall_13;
  sc_out<int> wstall_14;
  sc_out<int> wstall_15;
  sc_out<int> wstall_16;

  sc_out<int> outS;
  sc_out<int> w1SS;
  sc_out<int> w2SS;
  sc_out<int> w3SS;
  sc_out<int> w4SS;
  sc_out<int> w5SS;
  sc_out<int> w6SS;
  sc_out<int> w7SS;
  sc_out<int> w8SS;
  sc_out<int> w9SS;
  sc_out<int> w10SS;
  sc_out<int> w11SS;
  sc_out<int> w12SS;
  sc_out<int> w13SS;
  sc_out<int> w14SS;
  sc_out<int> w15SS;
  sc_out<int> w16SS;

  sc_out<int> schS;
  sc_out<int> p1S;

#if VMM_COUNT == 1
  struct var_array1 vars;
#elif VMM_COUNT == 2
  struct var_array2 vars;
#elif VMM_COUNT == 4
  struct var_array4 vars;
#elif VMM_COUNT == 6
  struct var_array6 vars;
#elif VMM_COUNT == 8
  struct var_array8 vars;
#elif VMM_COUNT == 16
  struct var_array16 vars;
#else
  exit(10); // not supported
#endif // VMM_COUNT

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
  ClockCycles *idle5 = new ClockCycles("idle5", true);
  ClockCycles *idle6 = new ClockCycles("idle6", true);
  ClockCycles *idle7 = new ClockCycles("idle7", true);
  ClockCycles *idle8 = new ClockCycles("idle8", true);
  ClockCycles *idle9 = new ClockCycles("idle9", true);
  ClockCycles *idle10 = new ClockCycles("idle10", true);
  ClockCycles *idle11 = new ClockCycles("idle11", true);
  ClockCycles *idle12 = new ClockCycles("idle12", true);
  ClockCycles *idle13 = new ClockCycles("idle13", true);
  ClockCycles *idle14 = new ClockCycles("idle14", true);
  ClockCycles *idle15 = new ClockCycles("idle15", true);
  ClockCycles *idle16 = new ClockCycles("idle16", true);

  ClockCycles *post1 = new ClockCycles("post1", true);
  ClockCycles *post2 = new ClockCycles("post2", true);
  ClockCycles *post3 = new ClockCycles("post3", true);
  ClockCycles *post4 = new ClockCycles("post4", true);
  ClockCycles *post5 = new ClockCycles("post5", true);
  ClockCycles *post6 = new ClockCycles("post6", true);
  ClockCycles *post7 = new ClockCycles("post7", true);
  ClockCycles *post8 = new ClockCycles("post8", true);
  ClockCycles *post9 = new ClockCycles("post9", true);
  ClockCycles *post10 = new ClockCycles("post10", true);
  ClockCycles *post11 = new ClockCycles("post11", true);
  ClockCycles *post12 = new ClockCycles("post12", true);
  ClockCycles *post13 = new ClockCycles("post13", true);
  ClockCycles *post14 = new ClockCycles("post14", true);
  ClockCycles *post15 = new ClockCycles("post15", true);
  ClockCycles *post16 = new ClockCycles("post16", true);

  ClockCycles *gemm1 = new ClockCycles("gemm1", true);
  ClockCycles *gemm2 = new ClockCycles("gemm2", true);
  ClockCycles *gemm3 = new ClockCycles("gemm3", true);
  ClockCycles *gemm4 = new ClockCycles("gemm4", true);
  ClockCycles *gemm5 = new ClockCycles("gemm5", true);
  ClockCycles *gemm6 = new ClockCycles("gemm6", true);
  ClockCycles *gemm7 = new ClockCycles("gemm7", true);
  ClockCycles *gemm8 = new ClockCycles("gemm8", true);
  ClockCycles *gemm9 = new ClockCycles("gemm9", true);
  ClockCycles *gemm10 = new ClockCycles("gemm10", true);
  ClockCycles *gemm11 = new ClockCycles("gemm11", true);
  ClockCycles *gemm12 = new ClockCycles("gemm12", true);
  ClockCycles *gemm13 = new ClockCycles("gemm13", true);
  ClockCycles *gemm14 = new ClockCycles("gemm14", true);
  ClockCycles *gemm15 = new ClockCycles("gemm15", true);
  ClockCycles *gemm16 = new ClockCycles("gemm16", true);

  ClockCycles *wstall1 = new ClockCycles("wstall1", true);
  ClockCycles *wstall2 = new ClockCycles("wstall2", true);
  ClockCycles *wstall3 = new ClockCycles("wstall3", true);
  ClockCycles *wstall4 = new ClockCycles("wstall4", true);
  ClockCycles *wstall5 = new ClockCycles("wstall5", true);
  ClockCycles *wstall6 = new ClockCycles("wstall6", true);
  ClockCycles *wstall7 = new ClockCycles("wstall7", true);
  ClockCycles *wstall8 = new ClockCycles("wstall8", true);
  ClockCycles *wstall9 = new ClockCycles("wstall9", true);
  ClockCycles *wstall10 = new ClockCycles("wstall10", true);
  ClockCycles *wstall11 = new ClockCycles("wstall11", true);
  ClockCycles *wstall12 = new ClockCycles("wstall12", true);
  ClockCycles *wstall13 = new ClockCycles("wstall13", true);
  ClockCycles *wstall14 = new ClockCycles("wstall14", true);
  ClockCycles *wstall15 = new ClockCycles("wstall15", true);
  ClockCycles *wstall16 = new ClockCycles("wstall16", true);

  BufferSpace *ginputbuf_p = new BufferSpace("ginputbuf_p", GINP_BUF_LEN);
  BufferSpace *inputbuf_p = new BufferSpace("inputbuf_p", INP_BUF_LEN);
  BufferSpace *weightbuf_p = new BufferSpace("weightbuf_p", WGT_BUF_LEN);
  DataCountArray *gmacs = new DataCountArray("gmacs", 4);
  DataCountArray *gouts = new DataCountArray("gouts", 4);

  // SignalTrack *shS = new SignalTrack("shS", true);
  // SignalTrack *gmSA = new SignalTrack("gmSA", true);
  // SignalTrack *gmSB = new SignalTrack("gmSB", true);
  // SignalTrack *gmSC = new SignalTrack("gmSC", true);
  // SignalTrack *gmSD = new SignalTrack("gmSD", true);
  // SignalTrack *psSA = new SignalTrack("psSA", true);
  // SignalTrack *psSB = new SignalTrack("psSB", true);
  // SignalTrack *psSC = new SignalTrack("psSC", true);
  // SignalTrack *psSD = new SignalTrack("psSD", true);

  SignalTrack *shS = new SignalTrack("T_shS", true);
  SignalTrack *gmSA = new SignalTrack("T_gmSA", true);
  SignalTrack *gmSB = new SignalTrack("T_gmSB", true);
  SignalTrack *gmSC = new SignalTrack("T_gmSC", true);
  SignalTrack *gmSD = new SignalTrack("T_gmSD", true);
  SignalTrack *psSA = new SignalTrack("T_psSA", true);
  SignalTrack *psSB = new SignalTrack("T_psSB", true);
  SignalTrack *psSC = new SignalTrack("T_psSC", true);
  SignalTrack *psSD = new SignalTrack("T_psSD", true);

  // std::vector<Metric *> profiling_vars = {shS,  gmSA, gmSB, gmSC, gmSD,
  //                                         psSA, psSB, psSC, psSD};
  // std::vector<Metric *> profiling_vars = {shS, gmSA, gmSB, psSA, psSB};
  std::vector<Metric *> profiling_vars = {shS, gmSB, psSB};

  // std::vector<Metric *> profiling_vars = {
  //     cycles, load_inps,    load_wgts,  compute,     idle1,   idle2,   idle3,
  //     idle4,  post1,        post2,      post3,       post4,   gemm1,   gemm2,
  //     gemm3,  gemm4,        wstall1,    wstall2,     wstall3, wstall4, shS,
  //     gmSA,   gmSB,         gmSC,       gmSD,        psSA,    psSB,    psSC,
  //     psSD,   gweightbuf_p, inputbuf_p, weightbuf_p, gmacs,   gouts};
#endif

  void init_VMM();

  void init_wgts_VMM();

  void wgt_len_VMM(unsigned int, unsigned int, unsigned int, unsigned int);

  void wgt_len_VMM_arr2(unsigned int, unsigned int[VMM_COUNT], bool[VMM_COUNT],
                        unsigned int);

  void wsum_len_VMM_arr2(unsigned int[VMM_COUNT], bool[VMM_COUNT]);

  void wgt_len_VMM_arr(unsigned int, unsigned int[VMM_COUNT], unsigned int);

  void wsum_len_VMM_arr(unsigned int[VMM_COUNT]);

  void wsum_len_VMM(unsigned int, unsigned int);

  void wgt_VMM_write_enable(int);

  void inp_len_VMM(unsigned int);

  void fill_wgts_VMM(sc_bigint<32 * 4>);

  void fill_wgts_VMM_individually(sc_bigint<32 * 4>, int);

  void fill_inps_VMM(sc_bigint<32 * 4>);

  void fill_wsums_VMM_all(sc_bigint<32 * 4>);

  void fill_wsums_VMM_individually(sc_bigint<32 * 4>, int);

  void fill_crf_VMM_all(sc_bigint<32 * 4>);

  void fill_crf_VMM_individually(sc_bigint<32 * 4>, int);

  void fill_crx_VMM_all(sc_int<32>);

  void fill_crx_VMM_individually(sc_int<32>, int);

  void wait_ready_VMM();

  void start_compute_VMM(unsigned int, unsigned int, unsigned int,
                         unsigned int);

  void send_done_write_VMM(int);

  void vmm_ready_VMM();

  void post_ready_VMM();

  void ppu_done_VMM();

  void Input_Handler();

  void Output_Handler();

  void Data_In();

  void Tracker();

  void Scheduler();

  void Arranger();

  void load_inputs(int, int);

  void schedule_vmm_unit(int, int, int, int, int);

  int SHR(int, int);

  void out_dbg(bool, bool, bool, bool);

  void VM_PE(ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
             ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
             ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
             ACC_DTYPE<32>[][4], int, int, int, int);

  void VM_PE_2(ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
               ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
               ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32>[][4], int, int);

  void VM_PE_DSP(ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
                 ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
                 ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32>[][4], int,
                 int);

  // void start_VMM(int, int, int, int[13]);
  void start_VMM(int, int, int, int);

#ifndef __SYNTHESIS__
  void Read_Cycle_Counter();

  void Writer_Cycle_Counter();
#endif

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_), arranger_fifo(16) {

    // Connect PE ports
    vars.init(clock, reset);

    SC_CTHREAD(Input_Handler, clock);
    reset_signal_is(reset, true);

    // SC_CTHREAD(Output_Handler, clock);
    // reset_signal_is(reset, true);

    SC_CTHREAD(Data_In, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Scheduler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Arranger, clock);
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(Read_Cycle_Counter, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Writer_Cycle_Counter, clock);
    reset_signal_is(reset, true);
#endif

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
