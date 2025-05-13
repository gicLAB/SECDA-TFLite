
#ifndef PE_MODULE_H
#define PE_MODULE_H

#include "acc_config.sc.h"

#define varsn(x) vars.vars_##x

SC_MODULE(PE) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_fifo_in<int> wgt_sum_fifo_in;
  sc_fifo_in<bUF> wgt_fifo_in;
  sc_fifo_in<bUF> inp_fifo_in;
  sc_fifo_out<ADATA> out_fifo_out;
  sc_fifo_in<int> temp_fifo_in;
  sc_fifo_out<int> temp_fifo_out;

  sc_fifo_in<ADATA> col_indices_fifo;
  sc_fifo_in<ADATA> out_indices_fifo;

  sc_in<bool> online;
  sc_in<bool> compute;
  sc_in<bool> reset_compute;
  sc_in<int> start_addr_p;
  sc_in<int> send_len_p;
  sc_in<int> bias_data;
  sc_in<int> crf_data;
  sc_in<int> crx_data;
  sc_in<int> ra_data;

  sc_in<bool> send;
  sc_in<int> cols_per_filter;
  sc_in<int> depth;
  sc_in<bool> process_cal;
  sc_in<int> oh;
  sc_in<int> ow;
  sc_in<int> kernel_size;
  sc_in<int> stride_x;
  sc_in<int> stride_y;
  sc_in<int> pt;
  sc_in<int> pl;
  sc_in<int> width_col;
  sc_in<int> crow;
  sc_in<int> num_rows;

  sc_out<bool> compute_done;
  sc_out<bool> wgt_loaded;
  sc_out<bool> send_done;
  sc_out<bool> process_cal_done;
  sc_out<int> computeS;
  sc_out<int> sendS;

  // sc_out_sig computeS;
  // sc_out_sig sendS;

  sc_int<16> depth_16;

  // Example for 3x3 kernel, has 9 gemm outputs, lets say only output 5, 6, 8
  // and 9 are used and the map to col2im out 0, 1, 2 and 3 respectively, then
  // col_indices = [5, 6, 8, 9] and out_indices = [0, 1, 2, 3]

  // pouts is the number of output needed to be computed using current
  // input row max value is ks * ks
  int pouts;

  int col_offset[PE_POUTDEXBUF_SIZE];
  int out_offset[PE_POUTDEXBUF_SIZE];

  // wgt_cols_buf needs to support ks * ks * depth / UF
  acc_dt wgt_cols_buf[PE_WGTCOLBUF_SIZE][UF];

  // wgt_col_sum needs to support ks * ks
  int wgt_col_sum[PE_WGTCOLSUMBUF_SIZE];

  // inp_row_buf needs to support depth / UF
  acc_dt inp_row_buf[PE_INPROWBUF_SIZE][UF];

  // Outbuf needs to support 1 * ks * ks gemm outputs
  int out_buf[PE_OUTBUF_SIZE];

  // Used to temporary accumalate GEMM outputs
  int acc_store[PE_ACC_BUF_SIZE];

  // temp inp and wgt buffers, supports UF unrolling
  acc_dt inp_temp[UF];

  int Quantised_Multiplier_gemmlowp(int x, int qm, sc_int<8> shift);

  int Quantised_Multiplier_ruy_reference(int x, int qm, sc_int<8> shift);

  void Compute();

  void Out();

  void Fill_Out();

  sc_int<32> mul_s8(sc_int<8> a, sc_int<8> b) {
    sc_int<32> c;
#pragma HLS RESOURCE variable = c core = Mul
    c = a * b;
    return c;
  }

  sc_int<64> mul_s64(int a, sc_int<64> b) {
    sc_int<64> c;
#pragma HLS RESOURCE variable = c core = Mul_LUT
    c = a * b;
    return c;
  }

  void init(sc_in<bool> & clock, sc_in<bool> & reset, PE_vars & vars) {
    this->clock(clock);
    this->reset(reset);
    this->computeS(vars.computeS);
    this->sendS(vars.sendS);

    this->wgt_sum_fifo_in(vars.wgt_sum_fifo);
    this->inp_fifo_in(vars.inp_fifo);
    this->wgt_fifo_in(vars.wgt_fifo);
    this->out_fifo_out(vars.out_fifo);
    this->temp_fifo_in(vars.temp_fifo);
    this->temp_fifo_out(vars.temp_fifo);
    this->col_indices_fifo(vars.col_indices_fifo);
    this->out_indices_fifo(vars.out_indices_fifo);

    this->online(vars.online);
    this->compute(vars.compute);
    this->reset_compute(vars.reset_compute);
    this->start_addr_p(vars.start_addr_p);
    this->send_len_p(vars.send_len_p);
    this->bias_data(vars.bias_data);
    this->crf_data(vars.crf_data);
    this->crx_data(vars.crx_data);
    this->ra_data(vars.ra_data);
    this->send(vars.send);
    this->cols_per_filter(vars.cols_per_filter);
    this->depth(vars.depth);
    this->process_cal(vars.process_cal);
    this->oh(vars.oh);
    this->ow(vars.ow);
    this->kernel_size(vars.kernel_size);
    this->stride_x(vars.stride_x);
    this->stride_y(vars.stride_y);
    this->pt(vars.pt);
    this->pl(vars.pl);
    this->width_col(vars.width_col);
    this->crow(vars.crow);
    this->num_rows(vars.num_rows);

    this->compute_done(vars.compute_done);
    this->wgt_loaded(vars.wgt_loaded);
    this->send_done(vars.send_done);
    this->process_cal_done(vars.process_cal_done);
  }

  SC_HAS_PROCESS(PE);

  PE(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Out, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Fill_Out, clock);
    reset_signal_is(reset, true);

    // SC_CTHREAD(Process_Cal_ID, clock.pos());
    // reset_signal_is(reset, true);

    // clang-format off
  #pragma HLS ARRAY_PARTITION variable = wgt_cols_buf cyclic factor = UF/2 dim = 2
  #pragma HLS ARRAY_PARTITION variable = inp_row_buf complete dim = 2
  #pragma HLS ARRAY_PARTITION variable = inp_temp complete

  #pragma HLS RESOURCE variable=col_offset core=RAM_2P_LUTRAM
  #pragma HLS RESOURCE variable=out_offset core=RAM_2P_LUTRAM
  #pragma HLS RESOURCE variable=out_buf core=RAM_2P_LUTRAM

    // clang-format on

    // #pragma HLS ARRAY_PARTITION variable = wgt_cols_buf complete dim = 2
    // #pragma HLS ARRAY_PARTITION variable = wgt_cols_buf cyclic factor = 2 dim
    // = 2
  }
};

// create var_array with 8 PEs
struct var_array {
  PE_vars vars_0;
  PE_vars vars_1;
  PE_vars vars_2;
  PE_vars vars_3;
  PE_vars vars_4;
  PE_vars vars_5;
  PE_vars vars_6;
  PE_vars vars_7;
  PE X1;
  PE X2;
  PE X3;
  PE X4;
  PE X5;
  PE X6;
  PE X7;
  PE X8;

#ifndef __SYNTHESIS__
  var_array()
      : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), vars_3(16, 3),
        vars_4(16, 4), vars_5(16, 5), vars_6(16, 6), vars_7(16, 7), X1("X1"),
        X2("X2"), X3("X3"), X4("X4"), X5("X5"), X6("X6"), X7("X7"), X8("X8") {}
#else
  var_array()
      : vars_0(16), vars_1(16), vars_2(16), vars_3(16), vars_4(16), vars_5(16),
        vars_6(16), vars_7(16), X1("X1"), X2("X2"), X3("X3"), X4("X4"),
        X5("X5"), X6("X6"), X7("X7"), X8("X8") {}
#endif

  PE_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else if (index == 1) return vars_1;
    else if (index == 2) return vars_2;
    else if (index == 3) return vars_3;
    else if (index == 4) return vars_4;
    else if (index == 5) return vars_5;
    else if (index == 6) return vars_6;
    else if (index == 7) return vars_7;
    else return vars_0;
  }

  void inp_write(bUF data, int index) {
    if (index == 0) return vars_0.inp_fifo.write(data);
    else if (index == 1) return vars_1.inp_fifo.write(data);
    else if (index == 2) return vars_2.inp_fifo.write(data);
    else if (index == 3) return vars_3.inp_fifo.write(data);
    else if (index == 4) return vars_4.inp_fifo.write(data);
    else if (index == 5) return vars_5.inp_fifo.write(data);
    else if (index == 6) return vars_6.inp_fifo.write(data);
    else if (index == 7) return vars_7.inp_fifo.write(data);
    else return vars_0.inp_fifo.write(data);
  }

  void col_indices_fifo_write(int d, bool tlast, int index) {
    ADATA data = {d, tlast};
    if (index == 0) return vars_0.col_indices_fifo.write(data);
    else if (index == 1) return vars_1.col_indices_fifo.write(data);
    else if (index == 2) return vars_2.col_indices_fifo.write(data);
    else if (index == 3) return vars_3.col_indices_fifo.write(data);
    else if (index == 4) return vars_4.col_indices_fifo.write(data);
    else if (index == 5) return vars_5.col_indices_fifo.write(data);
    else if (index == 6) return vars_6.col_indices_fifo.write(data);
    else if (index == 7) return vars_7.col_indices_fifo.write(data);
  }

  void out_indices_fifo_write(int d, bool tlast, int index) {
    ADATA data = {d, tlast};
    if (index == 0) vars_0.out_indices_fifo.write(data);
    if (index == 1) vars_1.out_indices_fifo.write(data);
    if (index == 2) vars_2.out_indices_fifo.write(data);
    if (index == 3) vars_3.out_indices_fifo.write(data);
    if (index == 4) vars_4.out_indices_fifo.write(data);
    if (index == 5) vars_5.out_indices_fifo.write(data);
    if (index == 6) vars_6.out_indices_fifo.write(data);
    if (index == 7) vars_7.out_indices_fifo.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) return vars_0.wgt_fifo.write(data);
    else if (index == 1) return vars_1.wgt_fifo.write(data);
    else if (index == 2) return vars_2.wgt_fifo.write(data);
    else if (index == 3) return vars_3.wgt_fifo.write(data);
    else if (index == 4) return vars_4.wgt_fifo.write(data);
    else if (index == 5) return vars_5.wgt_fifo.write(data);
    else if (index == 6) return vars_6.wgt_fifo.write(data);
    else if (index == 7) return vars_7.wgt_fifo.write(data);
  }
  void wgt_sum_fifo_write(int data, int index) {
    if (index == 0) return vars_0.wgt_sum_fifo.write(data);
    else if (index == 1) return vars_1.wgt_sum_fifo.write(data);
    else if (index == 2) return vars_2.wgt_sum_fifo.write(data);
    else if (index == 3) return vars_3.wgt_sum_fifo.write(data);
    else if (index == 4) return vars_4.wgt_sum_fifo.write(data);
    else if (index == 5) return vars_5.wgt_sum_fifo.write(data);
    else if (index == 6) return vars_6.wgt_sum_fifo.write(data);
    else if (index == 7) return vars_7.wgt_sum_fifo.write(data);
  }

  ADATA get(int index) {
    if (index == 0) return vars_0.out_fifo.read();
    else if (index == 1) return vars_1.out_fifo.read();
    else if (index == 2) return vars_2.out_fifo.read();
    else if (index == 3) return vars_3.out_fifo.read();
    else if (index == 4) return vars_4.out_fifo.read();
    else if (index == 5) return vars_5.out_fifo.read();
    else if (index == 6) return vars_6.out_fifo.read();
    else if (index == 7) return vars_7.out_fifo.read();
    return vars_0.out_fifo.read();
  };

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    X1.init(clock, reset, vars_0);
    X2.init(clock, reset, vars_1);
    X3.init(clock, reset, vars_2);
    X4.init(clock, reset, vars_3);
    X5.init(clock, reset, vars_4);
    X6.init(clock, reset, vars_5);
    X7.init(clock, reset, vars_6);
    X8.init(clock, reset, vars_7);
  }
};

#endif // PE_MODULE_H
