
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
  sc_fifo_out<DATA> out_fifo_out;
  sc_fifo_in<int> temp_fifo_in;
  sc_fifo_out<int> temp_fifo_out;

  sc_fifo_in<DATA> col_indices_fifo;
  sc_fifo_in<DATA> out_indices_fifo;

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
  sc_in<bool> out;
  sc_in<int> cols_per_filter;
  sc_in<int> depth;

  sc_out<bool> compute_done;
  sc_out<bool> wgt_loaded;
  sc_out<bool> out_done;
  sc_out<bool> send_done;

  sc_in<bool> process_cal;
  sc_out<bool> process_cal_done;

  // sc_out<int> computeS;
  // sc_out<int> sendS;

  sc_out_sig computeS;
  sc_out_sig sendS;

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

  // Example for 3x3 kernel, has 9 gemm outputs, lets say only output 5, 6, 8
  // and 9 are used and the map to col2im out 0, 1, 2 and 3 respectively, then
  // col_indices = [5, 6, 8, 9] and out_indices = [0, 1, 2, 3]

  // pouts is the number of output needed to be computed using current
  // input row max value is ks * ks
  int pouts;

  // col_indices is the indexes of the output needed to be computed using
  // current input row
  // int col_indices[PE_POUTDEXBUF_SIZE];
  int col_offset[PE_POUTDEXBUF_SIZE];

  // for each of those output cols, this indicates the col2im output mapping
  // int out_indices[PE_POUTDEXBUF_SIZE];

  // // wgt_cols_buf needs to support ks * ks * depth / UF
  acc_dt wgt_cols_buf[PE_WGTCOLBUF_SIZE][UF];

  // wgt_col_sum needs to support ks * ks
  int wgt_col_sum[PE_WGTCOLSUMBUF_SIZE];

  // single row of input , 32 is a limiting factor
  // x rows of weights (x * depth) (x = ks * ks)
  // inp_row_buf needs to support depth / UF
  acc_dt inp_row_buf[PE_INPROWBUF_SIZE][UF];

  // Outbuf needs to support ir * ks * ks gemm outputs where ir is the number
  // of input rows
  int out_buf[PE_OUTBUF_SIZE];

  // temp inp and wgt buffers, supports UF unrolling
  acc_dt inp_temp[UF];

  // Used to temporary accumalate GEMM outputs
  int acc_store[PE_ACC_BUF_SIZE];

#ifndef __SYNTHESIS__
  // Reference CPU
  int Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
    sc_int<64> pl;
    sc_int<32> pr;
    sc_int<32> msk;
    sc_int<32> sm;
    if (shift > 0) {
      pl = shift;
      pr = 0;
      msk = 0;
      sm = 0;
    } else {
      // pl = 1;
      pl = 0;
      pr = -shift;
      msk = (1 << -shift) - 1;
      sm = msk >> 1;
    }
    sc_int<64> val = x * (1 << pl);
    if (val > MAX) val = MAX;
    if (val < MIN) val = MIN;
    sc_int<64> val_2 = val * qm;
    sc_int<32> temp_1;
    temp_1 = (val_2 + POS) / DIVMAX;
    if (val_2 < 0) temp_1 = (val_2 + NEG) / DIVMAX;
    sc_int<32> val_3 = temp_1;
    val_3 = val_3 >> pr;
    sc_int<32> temp_2 = temp_1 & msk;
    sc_int<32> temp_3 = (temp_1 < 0) & 1;
    sc_int<32> temp_4 = sm + temp_3;
    sc_int<32> temp_5 = ((temp_2 > temp_4) & 1);
    sc_int<32> result_32 = val_3 + temp_5;
    int res = result_32;
    return result_32;
  }
#else
  //  ARM-Neon version CPU
  int Quantised_Multiplier(int x, int qm, sc_int<8> shift) {
    int nshift = shift;
    int total_shift = 31 - shift;
    sc_int<64> x_64 = x;
    sc_int<64> quantized_multiplier_64(qm);
    sc_int<64> one = 1;
    sc_int<64> round = one << (total_shift - 1); // ALU ADD + ALU SHLI
    sc_int<64> result =
        x_64 * quantized_multiplier_64 + round; // ALU ADD + ALU MUL
    result = result >> total_shift;             // ALU SHRI
    int nresult = result;
    if (result > MAX) result = MAX; // ALU MIN
    if (result < MIN) result = MIN; // ALU MAX
    sc_int<32> result_32 = result;
    return result_32;
  }
#endif

  void Compute() {
    compute_done.write(false);
    wgt_loaded.write(false);
    computeS.write(0);
    wait();
    while (1) {

      computeS.write(1);
      wgt_loaded.write(false);
      DWAIT();
      while (!online.read()) {
        wait();
      }

      // load weights
      int i = 0;
      computeS.write(2);
      for (int c = 0; c < cols_per_filter; c++) {
        wgt_col_sum[c] = wgt_sum_fifo_in.read();
        DWAIT(2);
        for (int d = 0; d < depth; d++) {
          bUF data = wgt_fifo_in.read();
          data.retreive(wgt_cols_buf, i);
          DWAIT();
          i++;
        }
      }
      wait();
      computeS.write(3);
      wgt_loaded.write(true);
      wait();

      computeS.write(4);
      // PE is active (activate_PEs() called)
      while (online) {
        computeS.write(5);

        // waiting for start_compute() call
        while (!compute) {
          computeS.write(51);
          if (!online) break;
          wait();
        }
        if (!online) break;

        // computeS.write(6);
        // loads inputs
        for (int d = 0; d < depth; d++) {
#pragma HLS PIPELINE II = 1
          bUF data = inp_fifo_in.read();
          data.retreive(inp_row_buf, d);
          DWAIT();
        }
        DWAIT(3);

        computeS.write(61);
        DWAIT();
        int pouts = 0;
        DATA d = col_indices_fifo.read();
        while (!d.tlast) {
#pragma HLS PIPELINE II = 1
          col_offset[pouts] = d.data * depth;
          out_buf[pouts++] += wgt_col_sum[d.data];
          d = col_indices_fifo.read();
          // DWAIT(7);
          DWAIT();
        }
        DWAIT(8);

        // computeS.write(7);
        for (int d = 0; d < depth; d++) {
          // load input
          for (int u = 0; u < UF; u++) {
#pragma HLS UNROLL
            inp_temp[u] = inp_row_buf[d][u];
          }
          DWAIT(2);
          for (int i = 0; i < pouts; i++) {
#pragma HLS loop_tripcount min = 20 max = 20 avg = 20
#pragma HLS PIPELINE II = 1
            int col_off = col_offset[i];
            int sum = 0;
            for (int u = 0; u < UF; u++) {
#pragma HLS UNROLL
              acc_dt wt1 = wgt_cols_buf[col_off + d][u];
              sum += wt1 * inp_temp[u];
            }
            out_buf[i] += sum;
            DWAIT();
          }
          DWAIT(9);
        }
        // computeS.write(71);
        DWAIT();
        computeS.write(6);
        for (int i = 0; i < pouts; i++) {
#pragma HLS PIPELINE II = 1
          int dout = out_buf[i];
          temp_fifo_out.write(out_buf[i]);
          DWAIT();
        }
        DWAIT(5);
        wait();
        for (int i = 0; i < pouts; i++) {
          out_buf[i] = 0;
          DWAIT();
        }

        computeS.write(8);
        compute_done.write(true);
        while (!reset_compute) {
          computeS.write(11);
          DWAIT();
        }

        compute_done.write(false);
        computeS.write(12);
      }
      computeS.write(13);
      DWAIT();
    }
  }

  void Out() {
    DATA last = {4000, 1};
    DATA d = {0, 0};
    int start_addr = 0;
    int send_len = 0;
    int bias;
    int crf;
    int ra = 0;
    out_done.write(false);
    send_done.write(false);
    sendS.write(0);
    sc_int<32> crx;

    wait();
    while (1) {
      while (!out.read() && !send.read()) {
        sendS.write(1);
        DWAIT();
      }
      sendS.write(2);
      if (send) {
        sendS.write(3);
        start_addr = start_addr_p.read();
        send_len = send_len_p.read();
        bias = bias_data.read();
        crf = crf_data.read();
        crx = crx_data.read();
        ra = ra_data.read();
        // sendS.write(31);
        // sendS.write(send_len);

        DWAIT(23);
        for (int i = 0; i < send_len; i++) {
#pragma HLS PIPELINE II = 1
          int dex = (start_addr + i) % PE_ACC_BUF_SIZE;
          int qm_ret = ra + Quantised_Multiplier(acc_store[dex] + bias, crf,
                                                 crx.range(7, 0));
          if (qm_ret > MAX8) qm_ret = MAX8;
          else if (qm_ret < MIN8) qm_ret = MIN8;
          d.data = qm_ret;
          // d.data = acc_store[dex];
          out_fifo_out.write(d);
          if (i + 1 == send_len) {
            out_fifo_out.write(last);
          }
          // sendS.write(32);
          DWAIT(2);
        }
        DWAIT();

        for (int i = 0; i < send_len; i++) {
          int dex = (start_addr + i) % PE_ACC_BUF_SIZE;
          acc_store[dex] = 0;
          DWAIT(4);
        }
        // sendS.write(33);
        send_done.write(true);
      }

      if (out) {
        sendS.write(4);
        DATA d = out_indices_fifo.read();
        while (!d.tlast) {
          int dex = d.data % PE_ACC_BUF_SIZE;
          int out_data = temp_fifo_in.read();
          acc_store[dex] += out_data;
          d = out_indices_fifo.read();
          DWAIT(6);
        }
        out_done.write(true);
      }

      sendS.write(5);
      wait();
      while (out || send) {
        sendS.write(6);
        DWAIT();
      }
      out_done.write(false);
      send_done.write(false);
      DWAIT();
    }
  }

  void init(sc_in<bool> & clock, sc_in<bool> & reset, PE_vars & vars) {
    this->clock(clock);
    this->reset(reset);
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
    this->out(vars.out);
    this->cols_per_filter(vars.cols_per_filter);
    this->depth(vars.depth);
    this->compute_done(vars.compute_done);
    this->wgt_loaded(vars.wgt_loaded);
    this->out_done(vars.out_done);
    this->send_done(vars.send_done);
    this->computeS(vars.computeS);
    this->sendS(vars.sendS);

    this->process_cal(vars.process_cal);
    this->process_cal_done(vars.process_cal_done);

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
  }

  SC_HAS_PROCESS(PE);

  PE(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Out, clock);
    reset_signal_is(reset, true);

    // SC_CTHREAD(Process_Cal_ID, clock.pos());
    // reset_signal_is(reset, true);

#pragma HLS ARRAY_PARTITION variable = wgt_cols_buf cyclic factor = 8 dim = 2
#pragma HLS ARRAY_PARTITION variable = inp_row_buf complete dim = 2
#pragma HLS ARRAY_PARTITION variable = inp_temp complete
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
    DATA data = {d, tlast};
    if (index == 0) return vars_0.col_indices_fifo.write(data);
    else if (index == 1) return vars_1.col_indices_fifo.write(data);
    else if (index == 2) return vars_2.col_indices_fifo.write(data);
    else if (index == 3) return vars_3.col_indices_fifo.write(data);
    else if (index == 4) return vars_4.col_indices_fifo.write(data);
    else if (index == 5) return vars_5.col_indices_fifo.write(data);
    else if (index == 6) return vars_6.col_indices_fifo.write(data);
    else if (index == 7) return vars_7.col_indices_fifo.write(data);
    else return vars_0.col_indices_fifo.write(data);
  }

  void out_indices_fifo_write(int d, bool tlast, int index) {
    DATA data = {d, tlast};
    if (index == 0) return vars_0.out_indices_fifo.write(data);
    else if (index == 1) return vars_1.out_indices_fifo.write(data);
    else if (index == 2) return vars_2.out_indices_fifo.write(data);
    else if (index == 3) return vars_3.out_indices_fifo.write(data);
    else if (index == 4) return vars_4.out_indices_fifo.write(data);
    else if (index == 5) return vars_5.out_indices_fifo.write(data);
    else if (index == 6) return vars_6.out_indices_fifo.write(data);
    else if (index == 7) return vars_7.out_indices_fifo.write(data);
    else return vars_0.out_indices_fifo.write(data);
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
    else return vars_0.wgt_fifo.write(data);
  }

  DATA get(int index) {
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
