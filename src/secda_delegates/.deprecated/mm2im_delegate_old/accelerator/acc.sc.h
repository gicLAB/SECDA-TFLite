#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include "pe_module.sc.h"

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_out<bool> on;

  sc_out<int> inS;
  sc_out<int> data_loadS;
  sc_out<int> scheduleS;
  sc_out<int> outS;
  sc_out<int> pdS;
  sc_out<int> tempS;

  // sc_out_sig inS;
  // sc_out_sig data_loadS;
  // sc_out_sig scheduleS;
  // sc_out_sig outS;
  // sc_out_sig tempS;

  sc_fifo_in<DATA> din1;
  sc_fifo_in<DATA> din2;
  sc_fifo_in<DATA> din3;
  sc_fifo_in<DATA> din4;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

  // Global Buffer for wgt and inp data
  // ACC_DTYPE<8> wgt_buf[WGT_BUF_LEN][UF];
  // ACC_DTYPE<8> inp_buf[INP_BUF_LEN][UF];

  // Global Buffer for wgt sum: needs to support ks * ks * filter_step
  ACC_DTYPE<32> wgt_sum_buf[G_WGTSUMBUF_SIZE];
  // struct sbram<ACC_DTYPE<32>,G_WGTSUMBUF_SIZE> wgt_sum_buf;

  // Global Buffer for bias, crf, crx:  needs to support filter_step
  ACC_DTYPE<32> bias_buf[PE_COUNT];
  ACC_DTYPE<32> crf_buf[PE_COUNT];
  ACC_DTYPE<32> crx_buf[PE_COUNT];
  int pe_cols[PE_COUNT];

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> load_wgt;
  sc_signal<bool, SC_MANY_WRITERS> load_inp;
  sc_signal<bool, SC_MANY_WRITERS> load_map;
  sc_signal<bool, SC_MANY_WRITERS> load_col_map;
  sc_signal<bool, SC_MANY_WRITERS> load_data;
  sc_signal<bool, SC_MANY_WRITERS> data_in;
  sc_signal<bool, SC_MANY_WRITERS> schedule;
  sc_signal<bool, SC_MANY_WRITERS> load_fifo;
  sc_signal<bool, SC_MANY_WRITERS> start_decode;
#else
  sc_signal<bool> load_wgt;
  sc_signal<bool> load_inp;
  sc_signal<bool> load_map;
  sc_signal<bool> load_col_map;
  sc_signal<bool> load_data;
  sc_signal<bool> data_in;
  sc_signal<bool> schedule;
  sc_signal<bool> load_fifo;
  sc_signal<bool> start_decode;
#endif

  int row_size;
  int depth;
  int cols_per_filter;
  int inp_rows;
  int number_of_rows;
  int nfilters;

  int ra;

  // Pattern vars
  int oh;
  int ow;
  int kernel_size;
  int stride_x;
  int stride_y;
  int pt;
  int pl;
  int width_col;
  int srow;

  struct var_array vars;

#ifndef __SYNTHESIS__
  // Profiling variable
  ClockCycles *schedule_cycles = new ClockCycles("schedule_cycles", true);
  ClockCycles *process_cycles = new ClockCycles("process_cycles", true);
  ClockCycles *store_cycles = new ClockCycles("store_cycles", true);
  ClockCycles *update_wgt_cycles = new ClockCycles("update_wgt_cycles", true);
  ClockCycles *update_inp_cycles = new ClockCycles("update_inp_cycles", true);

  ClockCycles *compute_cycles = new ClockCycles("compute_cycles", true);
  ClockCycles *send_cycles = new ClockCycles("send_cycles", true);
  ClockCycles *out_cycles = new ClockCycles("out_cycles", true);

  ClockCycles *load_wgt_cycles = new ClockCycles("load_wgt_cycles", true);
  ClockCycles *load_inp_cycles = new ClockCycles("load_inp_cycles", true);
  ClockCycles *load_col_map_cycles =
      new ClockCycles("load_col_map_cycles", true);

  SignalTrack *T_in = new SignalTrack("T_Input-Handler", true);
  SignalTrack *T_sh = new SignalTrack("T_Scheduler", true);
  SignalTrack *T_ld = new SignalTrack("T_Data-Loader", true);
  SignalTrack *T_com = new SignalTrack("T_PE-Compute", true);
  SignalTrack *T_sd = new SignalTrack("T_PE-Output", true);
  SignalTrack *T_pd = new SignalTrack("T_Pattern-Decoder", true);
  SignalTrack *T_var = new SignalTrack("T_Custom1", true);

  // ClockCycles *idle1 = new ClockCycles("idle1", true);
  // ClockCycles *idle2 = new ClockCycles("idle2", true);
  // BufferSpace *gweightbuf_p = new BufferSpace("gweightbuf_p", WGT_BUF_LEN);
  // BufferSpace *inputbuf_p = new BufferSpace("inputbuf_p", INP_BUF_LEN);
  // DataCountArray *gmacs = new DataCountArray("gmacs", 4);
  // DataCountArray *gouts = new DataCountArray("gouts", 4);

  // std::vector<Metric *> profiling_vars = {
  //     schedule_cycles,   process_cycles, store_cycles, update_wgt_cycles,
  //     update_inp_cycles, compute_cycles, send_cycles,  out_cycles,
  //     load_wgt_cycles,   load_inp_cycles};

  std::vector<Metric *> profiling_vars = {
      process_cycles, compute_cycles, T_in, T_sh, T_ld,
      T_com,          T_sd,           T_pd, T_var};

  void In_Counter();
#endif

  // Modules
  void Input_Handler();

  void Output_Handler();

  void Data_In();

  void Scheduler();

  void FIFO_Loader();

  void Pattern_Decoder();

  // Functions

  void init_PE_signals();

  void activate_PEs();

  void deactivate_PEs();

  void set_row_PEs(int);

  // void load_wgt_PEs();

  void load_inp_PEs();

  void store(int start, int length);

  bool wgt_loaded();

  bool compute_done();

  bool compute_resetted();

  bool store_done();

  void start_compute();

  void stop_compute();

  bool out_fifo_filled();

  void col_indices_write(int, bool);

  void out_indices_write(int, bool);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_), vars() {

    // Connect PE ports
    vars.init(clock, reset);

    SC_CTHREAD(Input_Handler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Output_Handler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Data_In, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Scheduler, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(FIFO_Loader, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Pattern_Decoder, clock);
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(In_Counter, clock);
    reset_signal_is(reset, true);
#endif

// #pragma HLS array_partition variable = wgt_buf dim = 2 cyclic factor = 8

// #pragma HLS array_partition variable = wgt_buf dim = 2 complete
// #pragma HLS array_partition variable = inp_buf dim = 2 complete

// #pragma HLS array_partition variable = ocols complete
#pragma HLS array_partition variable = pe_cols complete

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
