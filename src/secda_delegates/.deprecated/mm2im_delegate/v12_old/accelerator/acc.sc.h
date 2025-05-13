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

  sc_fifo_in<ADATA> din1;
  sc_fifo_in<ADATA> din2;
  sc_fifo_in<ADATA> din3;
  sc_fifo_in<ADATA> din4;

  sc_fifo_out<ADATA> dout1;
  sc_fifo_out<ADATA> dout2;
  sc_fifo_out<ADATA> dout3;
  sc_fifo_out<ADATA> dout4;

  // Global Buffer for bias, crf, crx:  needs to support filter_step
  ACC_DTYPE<32> bias_buf[PE_COUNT];
  ACC_DTYPE<32> crf_buf[PE_COUNT];
  ACC_DTYPE<32> crx_buf[PE_COUNT];
  double crx_scale_buf[PE_COUNT];


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
  sc_signal<bool, SC_MANY_WRITERS> start_out_decode;
  sc_signal<bool, SC_MANY_WRITERS> output_handler;
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
  sc_signal<bool> start_out_decode;
  sc_signal<bool> output_handler;
#endif

  int row_size;
  int depth;
  int cols_per_filter;
  int inp_rows;
  int number_of_rows;
  int nfilters;
  int send_len;
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

  // PE signal registers
  // Write registers
  bool online_reg[PE_COUNT];
  bool compute_reg[PE_COUNT];
  bool reset_compute_reg[PE_COUNT];
  int start_addr_p_reg[PE_COUNT];
  int send_len_p_reg[PE_COUNT];
  int bias_data_reg[PE_COUNT];
  int crf_data_reg[PE_COUNT];
  int crx_data_reg[PE_COUNT];
  double crx_scale_reg[PE_COUNT];
  int ra_data_reg[PE_COUNT];
  bool send_reg[PE_COUNT];
  int cols_per_filter_reg[PE_COUNT];
  int depth_reg[PE_COUNT];
  bool process_cal_reg[PE_COUNT];
  int oh_reg[PE_COUNT];
  int ow_reg[PE_COUNT];
  int kernel_size_reg[PE_COUNT];
  int stride_x_reg[PE_COUNT];
  int stride_y_reg[PE_COUNT];
  int pt_reg[PE_COUNT];
  int pl_reg[PE_COUNT];
  int width_col_reg[PE_COUNT];
  int crow_reg[PE_COUNT];
  int num_rows_reg[PE_COUNT];

  // Read registers
  bool compute_done_reg[PE_COUNT];
  bool wgt_loaded_reg[PE_COUNT];
  bool send_done_reg[PE_COUNT];
  bool process_cal_done_reg[PE_COUNT];

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
  SignalTrack *T_pd = new SignalTrack("T_Pattern-Decoder", true);

  SignalTrack *T_com_1 = new SignalTrack("T_PE-Compute-1", true);
  SignalTrack *T_com_2 = new SignalTrack("T_PE-Compute-2", true);
  SignalTrack *T_com_3 = new SignalTrack("T_PE-Compute-3", true);
  SignalTrack *T_com_4 = new SignalTrack("T_PE-Compute-4", true);
  SignalTrack *T_com_5 = new SignalTrack("T_PE-Compute-5", true);
  SignalTrack *T_com_6 = new SignalTrack("T_PE-Compute-6", true);
  SignalTrack *T_com_7 = new SignalTrack("T_PE-Compute-7", true);
  SignalTrack *T_com_8 = new SignalTrack("T_PE-Compute-8", true);

  SignalTrack *T_out_1 = new SignalTrack("T_PE-Out-1", true);
  SignalTrack *T_out_2 = new SignalTrack("T_PE-Out-2", true);
  SignalTrack *T_out_3 = new SignalTrack("T_PE-Out-3", true);
  SignalTrack *T_out_4 = new SignalTrack("T_PE-Out-4", true);
  SignalTrack *T_out_5 = new SignalTrack("T_PE-Out-5", true);
  SignalTrack *T_out_6 = new SignalTrack("T_PE-Out-6", true);
  SignalTrack *T_out_7 = new SignalTrack("T_PE-Out-7", true);
  SignalTrack *T_out_8 = new SignalTrack("T_PE-Out-8", true);

  SignalTrack *T_custom = new SignalTrack("T_Custom", true);


  std::vector<Metric *> profiling_vars = {
      process_cycles, compute_cycles, T_in,    T_sh,    T_ld,    T_pd,
      T_com_1,        T_com_2,        T_com_3, T_com_4, T_com_5, T_com_6,
      T_com_7,        T_com_8,        T_out_1, T_out_2, T_out_3, T_out_4,
      T_out_5,        T_out_6,        T_out_7, T_out_8, T_custom};

  // ClockCycles *idle1 = new ClockCycles("idle1", true);
  // ClockCycles *idle2 = new ClockCycles("idle2", true);
  // DataCountArray *gmacs = new DataCountArray("gmacs", 4);
  // DataCountArray *gouts = new DataCountArray("gouts", 4);

  // std::vector<Metric *> profiling_vars = {
  //     schedule_cycles,   process_cycles, store_cycles, update_wgt_cycles,
  //     update_inp_cycles, compute_cycles, send_cycles,  out_cycles,
  //     load_wgt_cycles,   load_inp_cycles};

  // std::vector<Metric *> profiling_vars = {
  //     process_cycles, compute_cycles, T_in, T_sh, T_ld,
  //     T_com,          T_sd,           T_pd};

  void In_Counter();
#endif

  // Modules
  void Input_Handler();

  void Output_Handler();

  void Data_In();

  void Scheduler();

  void FIFO_Loader();

  void Pattern_Decoder();

  void Output_Pattern_Decoder();


  // Functions
  void load_inp_PEs();

  void store(int start, int length);

  // PE Controls

  void init_PE_signals();

  void activate_PEs_lim(int);

  void deactivate_PEs_lim(int);

  bool wgt_loaded_lim(int);

  void start_compute_lim(int);

  void stop_compute_lim(int);

  bool compute_done_lim(int);

  bool compute_resetted_lim(int);

  bool store_done_lim(int);

  void col_indices_write_lim(int, bool, int);

  void out_indices_write_lim(int, bool, int);

  void col_out_indices_write_lim(int, int, bool, int);

  double double_from_bits(uint64_t w) {
    union {
      uint64_t as_bits;
      double as_value;
    } db64;
    db64.as_bits = w;
    return db64.as_value;
  }
  
  
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

// Write registers
#pragma HLS array_partition variable = online_reg complete
#pragma HLS array_partition variable = compute_reg complete
#pragma HLS array_partition variable = reset_compute_reg complete
#pragma HLS array_partition variable = start_addr_p_reg complete
#pragma HLS array_partition variable = send_len_p_reg complete
#pragma HLS array_partition variable = bias_data_reg complete
#pragma HLS array_partition variable = crf_data_reg complete
#pragma HLS array_partition variable = crx_data_reg complete
#pragma HLS array_partition variable = ra_data_reg complete
#pragma HLS array_partition variable = send_reg complete
#pragma HLS array_partition variable = cols_per_filter_reg complete
#pragma HLS array_partition variable = depth_reg complete
#pragma HLS array_partition variable = process_cal_reg complete
#pragma HLS array_partition variable = oh_reg complete
#pragma HLS array_partition variable = ow_reg complete
#pragma HLS array_partition variable = kernel_size_reg complete
#pragma HLS array_partition variable = stride_x_reg complete
#pragma HLS array_partition variable = stride_y_reg complete
#pragma HLS array_partition variable = pt_reg complete
#pragma HLS array_partition variable = pl_reg complete
#pragma HLS array_partition variable = width_col_reg complete
#pragma HLS array_partition variable = crow_reg complete
#pragma HLS array_partition variable = num_rows_reg complete

// Read registers
#pragma HLS array_partition variable = compute_done_reg complete
#pragma HLS array_partition variable = wgt_loaded_reg complete
#pragma HLS array_partition variable = send_done_reg complete
#pragma HLS array_partition variable = process_cal_done_reg complete

#pragma HLS array_partition variable = crf_buf complete
// #pragma HLS array_partition variable = crx_buf complete
// #pragma HLS array_partition variable = crx_scale_buf complete

#pragma HLS array_partition variable = bias_buf complete



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
