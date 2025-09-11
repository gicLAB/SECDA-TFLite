#ifndef VMM_UNIT_H
#define VMM_UNIT_H

#include "acc_config.sc.h"
#define vars_post_write(x, y) vars_##y.post_fifo.write(x)
SC_MODULE(VMM_UNIT) {
  // IO ports
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<bool> load_inp;
  sc_in<bool> load_wgt;
  sc_in<bool> load_wsum;
  sc_in<bool> compute;
  sc_in<bool> send_done;
  sc_out<bool> ready;
  sc_out<bool> vmm_ready;
  sc_out<bool> post_ready;
  sc_out<bool> ppu_done;

  sc_in<int> ra;
  sc_in<unsigned int> depth;
  sc_in<unsigned int> w_idx;
  sc_in<unsigned int> wsum_idx;
  sc_in<unsigned int> wgt_len;
  sc_in<unsigned int> wsum_len;
  sc_in<unsigned int> inp_len;
  sc_in<unsigned int> depthLoadWgt;
  sc_in<unsigned int> wgt_colno;

  // FIFOs
  sc_fifo_in<bUF> wgt_fifo;
  sc_fifo_in<bUF> inp_fifo;
  sc_fifo_in<bUF> wsum_fifo;
  sc_fifo_in<bUF> crf_fifo;
  sc_fifo_in<int> crx_fifo;
  sc_fifo_in<int> post_fifo;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

// Signals
#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> post_ready1;
#else
  sc_signal<bool> post_ready1;
#endif

  // Memory
  // ACC_DTYPE<32> inp_1a_1[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> inp_1b_1[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> inp_1c_1[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> inp_1d_1[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> inp_1a_1[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> inp_1b_1[INP_BUF_LEN / DEPTHTILE];

  // ACC_DTYPE<32> wgt_data1a[WGT_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data2a[WGT_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data3a[WGT_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data4a[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> wgt_data1a[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> wgt_data2a[WGT_BUF_LEN / DEPTHTILE];

  ACC_DTYPE<32> wgt_sum1[WSUMS_BUF_LEN];
  ACC_DTYPE<32> wgt_sum2[WSUMS_BUF_LEN];
  ACC_DTYPE<32> wgt_sum3[WSUMS_BUF_LEN];
  ACC_DTYPE<32> wgt_sum4[WSUMS_BUF_LEN];
  ACC_DTYPE<32> crf1[WSUMS_BUF_LEN];
  ACC_DTYPE<32> crf2[WSUMS_BUF_LEN];
  ACC_DTYPE<32> crf3[WSUMS_BUF_LEN];
  ACC_DTYPE<32> crf4[WSUMS_BUF_LEN];
  ACC_DTYPE<8> crx1[WSUMS_BUF_LEN];
  ACC_DTYPE<8> crx2[WSUMS_BUF_LEN];
  ACC_DTYPE<8> crx3[WSUMS_BUF_LEN];
  ACC_DTYPE<8> crx4[WSUMS_BUF_LEN];

  ACC_DTYPE<32> out[16][4];
  ACC_DTYPE<32> g[16];
  ACC_DTYPE<8> r[16];

#ifdef EN_DEPTHTILE
  // ACC_DTYPE<32> inp_1a_2[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> inp_1b_2[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> inp_1c_2[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> inp_1d_2[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> inp_1a_2[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> inp_1b_2[INP_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data1a_2[WGT_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data2a_2[WGT_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data3a_2[WGT_BUF_LEN / DEPTHTILE];
  // ACC_DTYPE<32> wgt_data4a_2[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> wgt_data1a_2[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<URAM_DATAWIDTH> wgt_data2a_2[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_2[16][4];

#if (DEPTHTILE == 4) || (DEPTHTILE == 6) || (DEPTHTILE == 8)
  ACC_DTYPE<32> inp_1a_3[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1b_3[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1c_3[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1d_3[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data1a_3[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data2a_3[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data3a_3[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data4a_3[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_3[16][4];

  ACC_DTYPE<32> inp_1a_4[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1b_4[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1c_4[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1d_4[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data1a_4[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data2a_4[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data3a_4[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data4a_4[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_4[16][4];

#if (DEPTHTILE == 6) || (DEPTHTILE == 8)
  ACC_DTYPE<32> inp_1a_5[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1b_5[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1c_5[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1d_5[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data1a_5[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data2a_5[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data3a_5[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data4a_5[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_5[16][4];

  ACC_DTYPE<32> inp_1a_6[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1b_6[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1c_6[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1d_6[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data1a_6[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data2a_6[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data3a_6[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data4a_6[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_6[16][4];
#if (DEPTHTILE == 8)
  ACC_DTYPE<32> inp_1a_7[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1b_7[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1c_7[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1d_7[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data1a_7[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data2a_7[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data3a_7[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data4a_7[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_7[16][4];

  ACC_DTYPE<32> inp_1a_8[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1b_8[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1c_8[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> inp_1d_8[INP_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data1a_8[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data2a_8[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data3a_8[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> wgt_data4a_8[WGT_BUF_LEN / DEPTHTILE];
  ACC_DTYPE<32> out_8[16][4];

#endif
#endif
#endif
#endif

  // Debug
  // sc_out_sig computeS;
  sc_out<int> computeS;
  sc_out<int> postS;

  // functions
  sc_int<16> mul_lut(sc_int<8>, sc_int<8>);

  sc_int<16> mul_dsp(sc_int<8>, sc_int<8>);

  sc_int<64> mul_s64(int, sc_int<64>);

  void VM_PE_URAM(ACC_DTYPE<URAM_DATAWIDTH> *, ACC_DTYPE<URAM_DATAWIDTH> *,
                  ACC_DTYPE<URAM_DATAWIDTH> *, ACC_DTYPE<URAM_DATAWIDTH> *,
                  ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
                  ACC_DTYPE<32> *, ACC_DTYPE<32>[][4], int, int, int, int);

  void VM_PE_DSP_URAM(ACC_DTYPE<URAM_DATAWIDTH> *, ACC_DTYPE<URAM_DATAWIDTH> *,
                      ACC_DTYPE<URAM_DATAWIDTH> *, ACC_DTYPE<URAM_DATAWIDTH> *,
                      ACC_DTYPE<32>[][4], int, int);

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

  int Quantised_Multiplier_gemmlowp(int, int, sc_int<64>, sc_int<32>,
                                    sc_int<32>, sc_int<32>);

  sc_int<64> Quantised_Multiplier_gemmlowp_part1(int x, int qm, sc_int<64> pl);

  int Quantised_Multiplier_gemmlowp_part2(sc_int<64> val_2, sc_int<32> pr,
                                          sc_int<32> msk, sc_int<32> sm);

  int Quantised_Multiplier_ruy_reference(int, int, sc_int<8>, sc_int<64>);

  void PPU(ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
           sc_int<8> *, sc_int<8> *, sc_int<8> *, sc_int<8> *, ACC_DTYPE<32> *,
           ACC_DTYPE<8> *, int);

  // modules
  void LoadWeights();

  void LoadWSumsCrfCrx();

  void LoadInputs();

  void Compute();

  void Post();

  void init(sc_in<bool> & clock, sc_in<bool> & reset, VMM_vars & vars) {
    this->clock(clock);
    this->reset(reset);
    this->load_inp(vars.load_inp);
    this->load_wgt(vars.load_wgt);
    this->load_wsum(vars.load_wsum);
    this->compute(vars.compute);
    this->send_done(vars.send_done);
    this->ready(vars.ready);
    this->vmm_ready(vars.vmm_ready);
    this->post_ready(vars.post_ready);
    this->ppu_done(vars.ppu_done);
    this->ra(vars.ra);
    this->depth(vars.depth);
    this->w_idx(vars.w_idx);
    this->wsum_idx(vars.wsum_idx);
    this->wgt_len(vars.wgt_len);
    this->wsum_len(vars.wsum_len);
    this->inp_len(vars.inp_len);
    this->depthLoadWgt(vars.depthLoadWgt);
    this->wgt_colno(vars.wgt_colno);
    this->wgt_fifo(vars.wgt_fifo);
    this->inp_fifo(vars.inp_fifo);
    this->wsum_fifo(vars.wsum_fifo);
    this->crf_fifo(vars.crf_fifo);
    this->crx_fifo(vars.crx_fifo);
    this->post_fifo(vars.post_fifo);
    this->dout1(vars.dout1);
    this->dout2(vars.dout2);
    this->dout3(vars.dout3);
    this->dout4(vars.dout4);
    this->computeS(vars.computeS);
    this->postS(vars.postS);
  }

  SC_HAS_PROCESS(VMM_UNIT);

  VMM_UNIT(sc_module_name name_) : sc_module(name_) {

    SC_CTHREAD(LoadWeights, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(LoadWSumsCrfCrx, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(LoadInputs, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Post, clock);
    reset_signal_is(reset, true);

#pragma HLS array_partition variable = out complete dim = 0

#ifdef EN_DEPTHTILE

// for depthwise tiling, DEPTHTILE == 2
#pragma HLS array_partition variable = out_2 complete dim = 0

#if (DEPTHTILE == 4) || (DEPTHTILE == 6) || (DEPTHTILE == 8)
#pragma HLS array_partition variable = out_3 complete dim = 0
#pragma HLS array_partition variable = out_4 complete dim = 0

#if (DEPTHTILE == 6) || (DEPTHTILE == 8)
#pragma HLS array_partition variable = out_5 complete dim = 0
#pragma HLS array_partition variable = out_6 complete dim = 0

#if (DEPTHTILE == 8)
#pragma HLS array_partition variable = out_7 complete dim = 0
#pragma HLS array_partition variable = out_8 complete dim = 0

#endif
#endif
#endif
#endif
#pragma HLS array_partition variable = g complete dim = 0
#pragma HLS array_partition variable = r complete dim = 0
  }
};

struct var_array1 {
  VMM_vars vars_0;
  VMM_UNIT V0;

#ifndef __SYNTHESIS__
  var_array1() : vars_0(16, 0), V0("V0") {}
#else
  var_array1() : vars_0(16), V0("V0") {}
#endif

  bool check_douts_empty(int index) {
    bool empty = true;
    if (index == 0)
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    else
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());

    return empty;
  }

  int next(int index) {
    if (index == 0) return 0;
    else return 0;
  }

  void post_write(int data, int index) {
    if (index == 0) vars_0.post_fifo.write(data);
    else vars_0.post_fifo.write(data);
  }

  void send_done_write(bool data, int index) {
    if (index == 0) vars_0.send_done.write(data);
    else vars_0.send_done.write(data);
  }

  void load_inp_write(bool data, int index) {
    if (index == 0) vars_0.load_inp.write(data);
    else vars_0.load_inp.write(data);
  }

  void inp_len_write(unsigned int len, int index) {
    if (index == 0) vars_0.inp_len.write(len);
    else vars_0.inp_len.write(len);
  }

  void inp_write(bUF data, int index) { vars_0.inp_fifo.write(data); }

  void load_wgt_write(bool data, int index) {
    if (index == 0) vars_0.load_wgt.write(data);
    else vars_0.load_wgt.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) vars_0.wgt_fifo.write(data);
    else vars_0.wgt_fifo.write(data);
  }

  void load_wsum_write(bool data, int index) {
    if (index == 0) vars_0.load_wsum.write(data);
    else vars_0.load_wsum.write(data);
  }

  void wsum_write(bUF data, int index) {
    if (index == 0) vars_0.wsum_fifo.write(data);
    else vars_0.wsum_fifo.write(data);
  }

  void crf_write(bUF data, int index) {
    if (index == 0) vars_0.crf_fifo.write(data);
    else vars_0.crf_fifo.write(data);
  }

  void crx_write(int data, int index) {
    if (index == 0) vars_0.crx_fifo.write(data);
    else vars_0.crx_fifo.write(data);
  }

  DATA dout_read(int index, int dout_index) {
    DATA d = {0, 0};
    if (index == 0 && dout_index == 0) return vars_0.dout1.read();
    else if (index == 0 && dout_index == 1) return vars_0.dout2.read();
    else if (index == 0 && dout_index == 2) return vars_0.dout3.read();
    else if (index == 0 && dout_index == 3) return vars_0.dout4.read();
    else return d;
  }

  bool check_ready(int index) {
    if (index == 0) return vars_0.ready.read();
    else return vars_0.ready.read();
  }

  bool check_vmm_ready(int index) {
    if (index == 0) return vars_0.vmm_ready.read();
    else return vars_0.vmm_ready.read();
  }

  bool check_post_ready(int index) {
    if (index == 0) return vars_0.post_ready.read();
    else return vars_0.post_ready.read();
  }

  bool check_ppu_done(int index) {
    if (index == 0) return vars_0.ppu_done.read();
    else return vars_0.ppu_done.read();
  }

  void start_compute(int index, unsigned int w_idx, unsigned int wsum_idx,
                     unsigned int depth, int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    }
  }

  void set_compute(int index, bool compute) {
    if (index == 0) vars_0.compute.write(compute);
    else vars_0.compute.write(compute);
  }

  VMM_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else return vars_0;
  }

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    V0.init(clock, reset, vars_0);
  }
};

struct var_array2 {
  VMM_vars vars_0;
  VMM_vars vars_1;
  VMM_UNIT V0;
  VMM_UNIT V1;

#ifndef __SYNTHESIS__
  var_array2() : vars_0(16, 0), vars_1(16, 1), V0("V0"), V1("V1") {}
#else
  var_array2() : vars_0(16), vars_1(16), V0("V0"), V1("V1") {}
#endif

  bool check_douts_empty(int index) {
    bool empty = true;
    if (index == 0)
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    else if (index == 1)
      empty = !(vars_1.dout1.num_available() || vars_1.dout2.num_available() ||
                vars_1.dout3.num_available() || vars_1.dout4.num_available());
    else
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());

    return empty;
  }

  int next(int index) {
    if (index == 0) return 1;
    else if (index == 1) return 0;
    else return 0;
  }

  void post_write(int data, int index) {
    if (index == 0) vars_0.post_fifo.write(data);
    else if (index == 1) vars_1.post_fifo.write(data);
    else vars_0.post_fifo.write(data);
  }

  void send_done_write(bool data, int index) {
    if (index == 0) vars_0.send_done.write(data);
    else if (index == 1) vars_1.send_done.write(data);
    else vars_0.send_done.write(data);
  }

  void load_inp_write(bool data, int index) {
    if (index == 0) vars_0.load_inp.write(data);
    else if (index == 1) vars_1.load_inp.write(data);
    else vars_0.load_inp.write(data);
  }

  void inp_len_write(unsigned int len, int index) {
    if (index == 0) vars_0.inp_len.write(len);
    else if (index == 1) vars_1.inp_len.write(len);
    else vars_0.inp_len.write(len);
  }

  void inp_write(bUF data, int index) {
    vars_0.inp_fifo.write(data);
    vars_1.inp_fifo.write(data);
  }

  void load_wgt_write(bool data, int index) {
    if (index == 0) vars_0.load_wgt.write(data);
    else if (index == 1) vars_1.load_wgt.write(data);
    else vars_0.load_wgt.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) vars_0.wgt_fifo.write(data);
    else if (index == 1) vars_1.wgt_fifo.write(data);
    else vars_0.wgt_fifo.write(data);
  }

  void load_wsum_write(bool data, int index) {
    if (index == 0) vars_0.load_wsum.write(data);
    else if (index == 1) vars_1.load_wsum.write(data);
    else vars_0.load_wsum.write(data);
  }

  void wsum_write(bUF data, int index) {
    if (index == 0) vars_0.wsum_fifo.write(data);
    else if (index == 1) vars_1.wsum_fifo.write(data);
    else vars_0.wsum_fifo.write(data);
  }

  void crf_write(bUF data, int index) {
    if (index == 0) vars_0.crf_fifo.write(data);
    else if (index == 1) vars_1.crf_fifo.write(data);
    else vars_0.crf_fifo.write(data);
  }

  void crx_write(int data, int index) {
    if (index == 0) vars_0.crx_fifo.write(data);
    else if (index == 1) vars_1.crx_fifo.write(data);
    else vars_0.crx_fifo.write(data);
  }

  DATA dout_read(int index, int dout_index) {
    DATA d = {0, 0};
    if (index == 0 && dout_index == 0) return vars_0.dout1.read();
    else if (index == 0 && dout_index == 1) return vars_0.dout2.read();
    else if (index == 0 && dout_index == 2) return vars_0.dout3.read();
    else if (index == 0 && dout_index == 3) return vars_0.dout4.read();
    else if (index == 1 && dout_index == 0) return vars_1.dout1.read();
    else if (index == 1 && dout_index == 1) return vars_1.dout2.read();
    else if (index == 1 && dout_index == 2) return vars_1.dout3.read();
    else if (index == 1 && dout_index == 3) return vars_1.dout4.read();
    else return d;
  }

  bool check_ready(int index) {
    if (index == 0) return vars_0.ready.read();
    else if (index == 1) return vars_1.ready.read();
    else return vars_0.ready.read();
  }

  bool check_vmm_ready(int index) {
    if (index == 0) return vars_0.vmm_ready.read();
    else if (index == 1) return vars_1.vmm_ready.read();
    else return vars_0.vmm_ready.read();
  }

  bool check_post_ready(int index) {
    if (index == 0) return vars_0.post_ready.read();
    else if (index == 1) return vars_1.post_ready.read();
    else return vars_0.post_ready.read();
  }

  bool check_ppu_done(int index) {
    if (index == 0) return vars_0.ppu_done.read();
    else if (index == 1) return vars_1.ppu_done.read();
    else return vars_0.ppu_done.read();
  }

  void start_compute(int index, unsigned int w_idx, unsigned int wsum_idx,
                     unsigned int depth, int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    } else if (index == 1) {
      vars_1.ra.write(ra);
      vars_1.depth.write(depth);
      vars_1.w_idx.write(w_idx);
      vars_1.wsum_idx.write(wsum_idx);
      vars_1.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    }
  }

  void set_compute(int index, bool compute) {
    if (index == 0) vars_0.compute.write(compute);
    else if (index == 1) vars_1.compute.write(compute);
    else vars_0.compute.write(compute);
  }

  VMM_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else if (index == 1) return vars_1;
    else return vars_0;
  }

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    V0.init(clock, reset, vars_0);
    V1.init(clock, reset, vars_1);
  }
};

struct var_array4 {
  VMM_vars vars_0;
  VMM_vars vars_1;
  VMM_vars vars_2;
  VMM_vars vars_3;
  VMM_UNIT V0;
  VMM_UNIT V1;
  VMM_UNIT V2;
  VMM_UNIT V3;

#ifndef __SYNTHESIS__
  var_array4()
      : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), vars_3(16, 3), V0("V0"),
        V1("V1"), V2("V2"), V3("V3") {}
#else
  var_array4()
      : vars_0(16), vars_1(16), vars_2(16), vars_3(16), V0("V0"), V1("V1"),
        V2("V2"), V3("V3") {}
#endif

  bool check_douts_empty(int index) {
    bool empty = true;
    if (index == 0)
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    else if (index == 1)
      empty = !(vars_1.dout1.num_available() || vars_1.dout2.num_available() ||
                vars_1.dout3.num_available() || vars_1.dout4.num_available());
    else if (index == 2)
      empty = !(vars_2.dout1.num_available() || vars_2.dout2.num_available() ||
                vars_2.dout3.num_available() || vars_2.dout4.num_available());
    else if (index == 3)
      empty = !(vars_3.dout1.num_available() || vars_3.dout2.num_available() ||
                vars_3.dout3.num_available() || vars_3.dout4.num_available());
    else
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());

    return empty;
  }

  int next(int index) {
    if (index == 0) return 1;
    else if (index == 1) return 2;
    else if (index == 2) return 3;
    else if (index == 3) return 0;
    else return 0;
  }

  void post_write(int data, int index) {
    if (index == 0) vars_0.post_fifo.write(data);
    else if (index == 1) vars_1.post_fifo.write(data);
    else if (index == 2) vars_2.post_fifo.write(data);
    else if (index == 3) vars_3.post_fifo.write(data);
    else vars_0.post_fifo.write(data);
  }

  void send_done_write(bool data, int index) {
    if (index == 0) vars_0.send_done.write(data);
    else if (index == 1) vars_1.send_done.write(data);
    else if (index == 2) vars_2.send_done.write(data);
    else if (index == 3) vars_3.send_done.write(data);
    else vars_0.send_done.write(data);
  }

  void load_inp_write(bool data, int index) {
    if (index == 0) vars_0.load_inp.write(data);
    else if (index == 1) vars_1.load_inp.write(data);
    else if (index == 2) vars_2.load_inp.write(data);
    else if (index == 3) vars_3.load_inp.write(data);
    else vars_0.load_inp.write(data);
  }

  void inp_len_write(unsigned int len, int index) {
    if (index == 0) vars_0.inp_len.write(len);
    else if (index == 1) vars_1.inp_len.write(len);
    else if (index == 2) vars_2.inp_len.write(len);
    else if (index == 3) vars_3.inp_len.write(len);
    else vars_0.inp_len.write(len);
  }

  void inp_write(bUF data, int index) {
    vars_0.inp_fifo.write(data);
    vars_1.inp_fifo.write(data);
    vars_2.inp_fifo.write(data);
    vars_3.inp_fifo.write(data);
  }

  void load_wgt_write(bool data, int index) {
    if (index == 0) vars_0.load_wgt.write(data);
    else if (index == 1) vars_1.load_wgt.write(data);
    else if (index == 2) vars_2.load_wgt.write(data);
    else if (index == 3) vars_3.load_wgt.write(data);
    else vars_0.load_wgt.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) vars_0.wgt_fifo.write(data);
    else if (index == 1) vars_1.wgt_fifo.write(data);
    else if (index == 2) vars_2.wgt_fifo.write(data);
    else if (index == 3) vars_3.wgt_fifo.write(data);
    else vars_0.wgt_fifo.write(data);
  }

  void load_wsum_write(bool data, int index) {
    if (index == 0) vars_0.load_wsum.write(data);
    else if (index == 1) vars_1.load_wsum.write(data);
    else if (index == 2) vars_2.load_wsum.write(data);
    else if (index == 3) vars_3.load_wsum.write(data);
    else vars_0.load_wsum.write(data);
  }

  void wsum_write(bUF data, int index) {
    if (index == 0) vars_0.wsum_fifo.write(data);
    else if (index == 1) vars_1.wsum_fifo.write(data);
    else if (index == 2) vars_2.wsum_fifo.write(data);
    else if (index == 3) vars_3.wsum_fifo.write(data);
    else vars_0.wsum_fifo.write(data);
  }

  void crf_write(bUF data, int index) {
    if (index == 0) vars_0.crf_fifo.write(data);
    else if (index == 1) vars_1.crf_fifo.write(data);
    else if (index == 2) vars_2.crf_fifo.write(data);
    else if (index == 3) vars_3.crf_fifo.write(data);
    else vars_0.crf_fifo.write(data);
  }

  void crx_write(int data, int index) {
    if (index == 0) vars_0.crx_fifo.write(data);
    else if (index == 1) vars_1.crx_fifo.write(data);
    else if (index == 2) vars_2.crx_fifo.write(data);
    else if (index == 3) vars_3.crx_fifo.write(data);
    else vars_0.crx_fifo.write(data);
  }

  DATA dout_read(int index, int dout_index) {
    DATA d = {0, 0};
    if (index == 0 && dout_index == 0) return vars_0.dout1.read();
    else if (index == 0 && dout_index == 1) return vars_0.dout2.read();
    else if (index == 0 && dout_index == 2) return vars_0.dout3.read();
    else if (index == 0 && dout_index == 3) return vars_0.dout4.read();
    else if (index == 1 && dout_index == 0) return vars_1.dout1.read();
    else if (index == 1 && dout_index == 1) return vars_1.dout2.read();
    else if (index == 1 && dout_index == 2) return vars_1.dout3.read();
    else if (index == 1 && dout_index == 3) return vars_1.dout4.read();
    else if (index == 2 && dout_index == 0) return vars_2.dout1.read();
    else if (index == 2 && dout_index == 1) return vars_2.dout2.read();
    else if (index == 2 && dout_index == 2) return vars_2.dout3.read();
    else if (index == 2 && dout_index == 3) return vars_2.dout4.read();
    else if (index == 3 && dout_index == 0) return vars_3.dout1.read();
    else if (index == 3 && dout_index == 1) return vars_3.dout2.read();
    else if (index == 3 && dout_index == 2) return vars_3.dout3.read();
    else if (index == 3 && dout_index == 3) return vars_3.dout4.read();
    else return d;
  }

  bool check_ready(int index) {
    if (index == 0) return vars_0.ready.read();
    else if (index == 1) return vars_1.ready.read();
    else if (index == 2) return vars_2.ready.read();
    else if (index == 3) return vars_3.ready.read();
    else return vars_0.ready.read();
  }

  bool check_vmm_ready(int index) {
    if (index == 0) return vars_0.vmm_ready.read();
    else if (index == 1) return vars_1.vmm_ready.read();
    else if (index == 2) return vars_2.vmm_ready.read();
    else if (index == 3) return vars_3.vmm_ready.read();
    else return vars_0.vmm_ready.read();
  }

  bool check_post_ready(int index) {
    if (index == 0) return vars_0.post_ready.read();
    else if (index == 1) return vars_1.post_ready.read();
    else if (index == 2) return vars_2.post_ready.read();
    else if (index == 3) return vars_3.post_ready.read();
    else return vars_0.post_ready.read();
  }

  bool check_ppu_done(int index) {
    if (index == 0) return vars_0.ppu_done.read();
    else if (index == 1) return vars_1.ppu_done.read();
    else if (index == 2) return vars_2.ppu_done.read();
    else if (index == 3) return vars_3.ppu_done.read();
    else return vars_0.ppu_done.read();
  }

  void start_compute(int index, unsigned int w_idx, unsigned int wsum_idx,
                     unsigned int depth, int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    } else if (index == 1) {
      vars_1.ra.write(ra);
      vars_1.depth.write(depth);
      vars_1.w_idx.write(w_idx);
      vars_1.wsum_idx.write(wsum_idx);
      vars_1.compute.write(true);
    } else if (index == 2) {
      vars_2.ra.write(ra);
      vars_2.depth.write(depth);
      vars_2.w_idx.write(w_idx);
      vars_2.wsum_idx.write(wsum_idx);
      vars_2.compute.write(true);
    } else if (index == 3) {
      vars_3.ra.write(ra);
      vars_3.depth.write(depth);
      vars_3.w_idx.write(w_idx);
      vars_3.wsum_idx.write(wsum_idx);
      vars_3.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    }
  }

  void set_compute(int index, bool compute) {
    if (index == 0) vars_0.compute.write(compute);
    else if (index == 1) vars_1.compute.write(compute);
    else if (index == 2) vars_2.compute.write(compute);
    else if (index == 3) vars_3.compute.write(compute);
    else vars_0.compute.write(compute);
  }

  VMM_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else if (index == 1) return vars_1;
    else if (index == 2) return vars_2;
    else if (index == 3) return vars_3;
    else return vars_0;
  }

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    V0.init(clock, reset, vars_0);
    V1.init(clock, reset, vars_1);
    V2.init(clock, reset, vars_2);
    V3.init(clock, reset, vars_3);
  }
};

struct var_array6 {
  VMM_vars vars_0;
  VMM_vars vars_1;
  VMM_vars vars_2;
  VMM_vars vars_3;
  VMM_vars vars_4;
  VMM_vars vars_5;
  VMM_UNIT V0;
  VMM_UNIT V1;
  VMM_UNIT V2;
  VMM_UNIT V3;
  VMM_UNIT V4;
  VMM_UNIT V5;

#ifndef __SYNTHESIS__
  var_array6()
      : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), vars_3(16, 3),
        vars_4(16, 4), vars_5(16, 5), V0("V0"), V1("V1"), V2("V2"), V3("V3"),
        V4("V4"), V5("V5") {}
#else
  var_array6()
      : vars_0(16), vars_1(16), vars_2(16), vars_3(16), vars_4(16), vars_5(16),
        V0("V0"), V1("V1"), V2("V2"), V3("V3"), V4("V4"), V5("V5") {}
#endif

  bool check_douts_empty(int index) {
    bool empty = true;
    if (index == 0)
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    else if (index == 1)
      empty = !(vars_1.dout1.num_available() || vars_1.dout2.num_available() ||
                vars_1.dout3.num_available() || vars_1.dout4.num_available());
    else if (index == 2)
      empty = !(vars_2.dout1.num_available() || vars_2.dout2.num_available() ||
                vars_2.dout3.num_available() || vars_2.dout4.num_available());
    else if (index == 3)
      empty = !(vars_3.dout1.num_available() || vars_3.dout2.num_available() ||
                vars_3.dout3.num_available() || vars_3.dout4.num_available());
    else if (index == 4)
      empty = !(vars_4.dout1.num_available() || vars_4.dout2.num_available() ||
                vars_4.dout3.num_available() || vars_4.dout4.num_available());
    else if (index == 5)
      empty = !(vars_5.dout1.num_available() || vars_5.dout2.num_available() ||
                vars_5.dout3.num_available() || vars_5.dout4.num_available());
    else
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    return empty;
  }

  int next(int index) {
    if (index == 0) return 1;
    else if (index == 1) return 2;
    else if (index == 2) return 3;
    else if (index == 3) return 4;
    else if (index == 4) return 5;
    else if (index == 5) return 0;
    else return 0;
  }

  void post_write(int data, int index) {
    if (index == 0) vars_0.post_fifo.write(data);
    else if (index == 1) vars_1.post_fifo.write(data);
    else if (index == 2) vars_2.post_fifo.write(data);
    else if (index == 3) vars_3.post_fifo.write(data);
    else if (index == 4) vars_4.post_fifo.write(data);
    else if (index == 5) vars_5.post_fifo.write(data);
    else vars_0.post_fifo.write(data);
  }

  void send_done_write(bool data, int index) {
    if (index == 0) vars_0.send_done.write(data);
    else if (index == 1) vars_1.send_done.write(data);
    else if (index == 2) vars_2.send_done.write(data);
    else if (index == 3) vars_3.send_done.write(data);
    else if (index == 4) vars_4.send_done.write(data);
    else if (index == 5) vars_5.send_done.write(data);
    else vars_0.send_done.write(data);
  }

  void load_inp_write(bool data, int index) {
    if (index == 0) vars_0.load_inp.write(data);
    else if (index == 1) vars_1.load_inp.write(data);
    else if (index == 2) vars_2.load_inp.write(data);
    else if (index == 3) vars_3.load_inp.write(data);
    else if (index == 4) vars_4.load_inp.write(data);
    else if (index == 5) vars_5.load_inp.write(data);
    else vars_0.load_inp.write(data);
  }

  void inp_len_write(unsigned int len, int index) {
    if (index == 0) vars_0.inp_len.write(len);
    else if (index == 1) vars_1.inp_len.write(len);
    else if (index == 2) vars_2.inp_len.write(len);
    else if (index == 3) vars_3.inp_len.write(len);
    else if (index == 4) vars_4.inp_len.write(len);
    else if (index == 5) vars_5.inp_len.write(len);
    else vars_0.inp_len.write(len);
  }

  void inp_write(bUF data, int index) {
    vars_0.inp_fifo.write(data);
    vars_1.inp_fifo.write(data);
    vars_2.inp_fifo.write(data);
    vars_3.inp_fifo.write(data);
    vars_4.inp_fifo.write(data);
    vars_5.inp_fifo.write(data);
  }

  void load_wgt_write(bool data, int index) {
    if (index == 0) vars_0.load_wgt.write(data);
    else if (index == 1) vars_1.load_wgt.write(data);
    else if (index == 2) vars_2.load_wgt.write(data);
    else if (index == 3) vars_3.load_wgt.write(data);
    else if (index == 4) vars_4.load_wgt.write(data);
    else if (index == 5) vars_5.load_wgt.write(data);
    else vars_0.load_wgt.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) vars_0.wgt_fifo.write(data);
    else if (index == 1) vars_1.wgt_fifo.write(data);
    else if (index == 2) vars_2.wgt_fifo.write(data);
    else if (index == 3) vars_3.wgt_fifo.write(data);
    else if (index == 4) vars_4.wgt_fifo.write(data);
    else if (index == 5) vars_5.wgt_fifo.write(data);
    else vars_0.wgt_fifo.write(data);
  }

  void load_wsum_write(bool data, int index) {
    if (index == 0) vars_0.load_wsum.write(data);
    else if (index == 1) vars_1.load_wsum.write(data);
    else if (index == 2) vars_2.load_wsum.write(data);
    else if (index == 3) vars_3.load_wsum.write(data);
    else if (index == 4) vars_4.load_wsum.write(data);
    else if (index == 5) vars_5.load_wsum.write(data);
    else vars_0.load_wsum.write(data);
  }

  void wsum_write(bUF data, int index) {
    if (index == 0) vars_0.wsum_fifo.write(data);
    else if (index == 1) vars_1.wsum_fifo.write(data);
    else if (index == 2) vars_2.wsum_fifo.write(data);
    else if (index == 3) vars_3.wsum_fifo.write(data);
    else if (index == 4) vars_4.wsum_fifo.write(data);
    else if (index == 5) vars_5.wsum_fifo.write(data);
    else vars_0.wsum_fifo.write(data);
  }

  void crf_write(bUF data, int index) {
    if (index == 0) vars_0.crf_fifo.write(data);
    else if (index == 1) vars_1.crf_fifo.write(data);
    else if (index == 2) vars_2.crf_fifo.write(data);
    else if (index == 3) vars_3.crf_fifo.write(data);
    else if (index == 4) vars_4.crf_fifo.write(data);
    else if (index == 5) vars_5.crf_fifo.write(data);
    else vars_0.crf_fifo.write(data);
  }

  void crx_write(int data, int index) {
    if (index == 0) vars_0.crx_fifo.write(data);
    else if (index == 1) vars_1.crx_fifo.write(data);
    else if (index == 2) vars_2.crx_fifo.write(data);
    else if (index == 3) vars_3.crx_fifo.write(data);
    else if (index == 4) vars_4.crx_fifo.write(data);
    else if (index == 5) vars_5.crx_fifo.write(data);
    else vars_0.crx_fifo.write(data);
  }

  DATA dout_read(int index, int dout_index) {
    DATA d = {0, 0};
    if (index == 0 && dout_index == 0) return vars_0.dout1.read();
    else if (index == 0 && dout_index == 1) return vars_0.dout2.read();
    else if (index == 0 && dout_index == 2) return vars_0.dout3.read();
    else if (index == 0 && dout_index == 3) return vars_0.dout4.read();
    else if (index == 1 && dout_index == 0) return vars_1.dout1.read();
    else if (index == 1 && dout_index == 1) return vars_1.dout2.read();
    else if (index == 1 && dout_index == 2) return vars_1.dout3.read();
    else if (index == 1 && dout_index == 3) return vars_1.dout4.read();
    else if (index == 2 && dout_index == 0) return vars_2.dout1.read();
    else if (index == 2 && dout_index == 1) return vars_2.dout2.read();
    else if (index == 2 && dout_index == 2) return vars_2.dout3.read();
    else if (index == 2 && dout_index == 3) return vars_2.dout4.read();
    else if (index == 3 && dout_index == 0) return vars_3.dout1.read();
    else if (index == 3 && dout_index == 1) return vars_3.dout2.read();
    else if (index == 3 && dout_index == 2) return vars_3.dout3.read();
    else if (index == 3 && dout_index == 3) return vars_3.dout4.read();
    else if (index == 4 && dout_index == 0) return vars_4.dout1.read();
    else if (index == 4 && dout_index == 1) return vars_4.dout2.read();
    else if (index == 4 && dout_index == 2) return vars_4.dout3.read();
    else if (index == 4 && dout_index == 3) return vars_4.dout4.read();
    else if (index == 5 && dout_index == 0) return vars_5.dout1.read();
    else if (index == 5 && dout_index == 1) return vars_5.dout2.read();
    else if (index == 5 && dout_index == 2) return vars_5.dout3.read();
    else if (index == 5 && dout_index == 3) return vars_5.dout4.read();
    else return d;
  }

  bool check_ready(int index) {
    if (index == 0) return vars_0.ready.read();
    else if (index == 1) return vars_1.ready.read();
    else if (index == 2) return vars_2.ready.read();
    else if (index == 3) return vars_3.ready.read();
    else if (index == 4) return vars_4.ready.read();
    else if (index == 5) return vars_5.ready.read();
    else return vars_0.ready.read();
  }

  bool check_vmm_ready(int index) {
    if (index == 0) return vars_0.vmm_ready.read();
    else if (index == 1) return vars_1.vmm_ready.read();
    else if (index == 2) return vars_2.vmm_ready.read();
    else if (index == 3) return vars_3.vmm_ready.read();
    else if (index == 4) return vars_4.vmm_ready.read();
    else if (index == 5) return vars_5.vmm_ready.read();
    else return vars_0.vmm_ready.read();
  }

  bool check_post_ready(int index) {
    if (index == 0) return vars_0.post_ready.read();
    else if (index == 1) return vars_1.post_ready.read();
    else if (index == 2) return vars_2.post_ready.read();
    else if (index == 3) return vars_3.post_ready.read();
    else if (index == 4) return vars_4.post_ready.read();
    else if (index == 5) return vars_5.post_ready.read();
    else return vars_0.post_ready.read();
  }

  bool check_ppu_done(int index) {
    if (index == 0) return vars_0.ppu_done.read();
    else if (index == 1) return vars_1.ppu_done.read();
    else if (index == 2) return vars_2.ppu_done.read();
    else if (index == 3) return vars_3.ppu_done.read();
    else if (index == 4) return vars_4.ppu_done.read();
    else if (index == 5) return vars_5.ppu_done.read();
    else return vars_0.ppu_done.read();
  }

  void start_compute(int index, unsigned int w_idx, unsigned int wsum_idx,
                     unsigned int depth, int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    } else if (index == 1) {
      vars_1.ra.write(ra);
      vars_1.depth.write(depth);
      vars_1.w_idx.write(w_idx);
      vars_1.wsum_idx.write(wsum_idx);
      vars_1.compute.write(true);
    } else if (index == 2) {
      vars_2.ra.write(ra);
      vars_2.depth.write(depth);
      vars_2.w_idx.write(w_idx);
      vars_2.wsum_idx.write(wsum_idx);
      vars_2.compute.write(true);
    } else if (index == 3) {
      vars_3.ra.write(ra);
      vars_3.depth.write(depth);
      vars_3.w_idx.write(w_idx);
      vars_3.wsum_idx.write(wsum_idx);
      vars_3.compute.write(true);
    } else if (index == 4) {
      vars_4.ra.write(ra);
      vars_4.depth.write(depth);
      vars_4.w_idx.write(w_idx);
      vars_4.wsum_idx.write(wsum_idx);
      vars_4.compute.write(true);
    } else if (index == 5) {
      vars_5.ra.write(ra);
      vars_5.depth.write(depth);
      vars_5.w_idx.write(w_idx);
      vars_5.wsum_idx.write(wsum_idx);
      vars_5.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    }
  }

  void set_compute(int index, bool compute) {
    if (index == 0) vars_0.compute.write(compute);
    else if (index == 1) vars_1.compute.write(compute);
    else if (index == 2) vars_2.compute.write(compute);
    else if (index == 3) vars_3.compute.write(compute);
    else if (index == 4) vars_4.compute.write(compute);
    else if (index == 5) vars_5.compute.write(compute);
    else vars_0.compute.write(compute);
  }

  VMM_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else if (index == 1) return vars_1;
    else if (index == 2) return vars_2;
    else if (index == 3) return vars_3;
    else if (index == 4) return vars_4;
    else if (index == 5) return vars_5;
    else return vars_0;
  }

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    V0.init(clock, reset, vars_0);
    V1.init(clock, reset, vars_1);
    V2.init(clock, reset, vars_2);
    V3.init(clock, reset, vars_3);
    V4.init(clock, reset, vars_4);
    V5.init(clock, reset, vars_5);
  }
};

struct var_array8 {
  VMM_vars vars_0;
  VMM_vars vars_1;
  VMM_vars vars_2;
  VMM_vars vars_3;
  VMM_vars vars_4;
  VMM_vars vars_5;
  VMM_vars vars_6;
  VMM_vars vars_7;
  VMM_UNIT V0;
  VMM_UNIT V1;
  VMM_UNIT V2;
  VMM_UNIT V3;
  VMM_UNIT V4;
  VMM_UNIT V5;
  VMM_UNIT V6;
  VMM_UNIT V7;

#ifndef __SYNTHESIS__
  var_array8()
      : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), vars_3(16, 3),
        vars_4(16, 4), vars_5(16, 5), vars_6(16, 6), vars_7(16, 7), V0("V0"),
        V1("V1"), V2("V2"), V3("V3"), V4("V4"), V5("V5"), V6("V6"), V7("V7") {}
#else
  var_array8()
      : vars_0(16), vars_1(16), vars_2(16), vars_3(16), vars_4(16), vars_5(16),
        vars_6(16), vars_7(16), V0("V0"), V1("V1"), V2("V2"), V3("V3"),
        V4("V4"), V5("V5"), V6("V6"), V7("V7") {}
#endif

  bool check_douts_empty(int index) {
    bool empty = true;
    if (index == 0)
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    else if (index == 1)
      empty = !(vars_1.dout1.num_available() || vars_1.dout2.num_available() ||
                vars_1.dout3.num_available() || vars_1.dout4.num_available());
    else if (index == 2)
      empty = !(vars_2.dout1.num_available() || vars_2.dout2.num_available() ||
                vars_2.dout3.num_available() || vars_2.dout4.num_available());
    else if (index == 3)
      empty = !(vars_3.dout1.num_available() || vars_3.dout2.num_available() ||
                vars_3.dout3.num_available() || vars_3.dout4.num_available());
    else if (index == 4)
      empty = !(vars_4.dout1.num_available() || vars_4.dout2.num_available() ||
                vars_4.dout3.num_available() || vars_4.dout4.num_available());
    else if (index == 5)
      empty = !(vars_5.dout1.num_available() || vars_5.dout2.num_available() ||
                vars_5.dout3.num_available() || vars_5.dout4.num_available());
    else if (index == 6)
      empty = !(vars_6.dout1.num_available() || vars_6.dout2.num_available() ||
                vars_6.dout3.num_available() || vars_6.dout4.num_available());
    else if (index == 7)
      empty = !(vars_7.dout1.num_available() || vars_7.dout2.num_available() ||
                vars_7.dout3.num_available() || vars_7.dout4.num_available());
    else
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    return empty;
  }

  int next(int index) {
    if (index == 0) return 1;
    else if (index == 1) return 2;
    else if (index == 2) return 3;
    else if (index == 3) return 4;
    else if (index == 4) return 5;
    else if (index == 5) return 6;
    else if (index == 6) return 7;
    else if (index == 7) return 0;
    else return 0;
  }

  void post_write(int data, int index) {
    if (index == 0) vars_0.post_fifo.write(data);
    else if (index == 1) vars_1.post_fifo.write(data);
    else if (index == 2) vars_2.post_fifo.write(data);
    else if (index == 3) vars_3.post_fifo.write(data);
    else if (index == 4) vars_4.post_fifo.write(data);
    else if (index == 5) vars_5.post_fifo.write(data);
    else if (index == 6) vars_6.post_fifo.write(data);
    else if (index == 7) vars_7.post_fifo.write(data);
    else vars_0.post_fifo.write(data);
  }

  void send_done_write(bool data, int index) {
    if (index == 0) vars_0.send_done.write(data);
    else if (index == 1) vars_1.send_done.write(data);
    else if (index == 2) vars_2.send_done.write(data);
    else if (index == 3) vars_3.send_done.write(data);
    else if (index == 4) vars_4.send_done.write(data);
    else if (index == 5) vars_5.send_done.write(data);
    else if (index == 6) vars_6.send_done.write(data);
    else if (index == 7) vars_7.send_done.write(data);
    else vars_0.send_done.write(data);
  }

  void load_inp_write(bool data, int index) {
    if (index == 0) vars_0.load_inp.write(data);
    else if (index == 1) vars_1.load_inp.write(data);
    else if (index == 2) vars_2.load_inp.write(data);
    else if (index == 3) vars_3.load_inp.write(data);
    else if (index == 4) vars_4.load_inp.write(data);
    else if (index == 5) vars_5.load_inp.write(data);
    else if (index == 6) vars_6.load_inp.write(data);
    else if (index == 7) vars_7.load_inp.write(data);
    else vars_0.load_inp.write(data);
  }

  void inp_len_write(unsigned int len, int index) {
    if (index == 0) vars_0.inp_len.write(len);
    else if (index == 1) vars_1.inp_len.write(len);
    else if (index == 2) vars_2.inp_len.write(len);
    else if (index == 3) vars_3.inp_len.write(len);
    else if (index == 4) vars_4.inp_len.write(len);
    else if (index == 5) vars_5.inp_len.write(len);
    else if (index == 6) vars_6.inp_len.write(len);
    else if (index == 7) vars_7.inp_len.write(len);
    else vars_0.inp_len.write(len);
  }

  void inp_write(bUF data, int index) {
    // if (index == 0) vars_0.inp_fifo.write(data);
    // else if (index == 1) vars_1.inp_fifo.write(data);
    // else if (index == 2) vars_2.inp_fifo.write(data);
    // else if (index == 3) vars_3.inp_fifo.write(data);
    // else if (index == 4) vars_4.inp_fifo.write(data);
    // else if (index == 5) vars_5.inp_fifo.write(data);
    // else if (index == 6) vars_6.inp_fifo.write(data);
    // else if (index == 7) vars_7.inp_fifo.write(data);
    // else vars_0.inp_fifo.write(data);
    vars_0.inp_fifo.write(data);
    vars_1.inp_fifo.write(data);
    vars_2.inp_fifo.write(data);
    vars_3.inp_fifo.write(data);
    vars_4.inp_fifo.write(data);
    vars_5.inp_fifo.write(data);
    vars_6.inp_fifo.write(data);
    vars_7.inp_fifo.write(data);
  }

  void load_wgt_write(bool data, int index) {
    if (index == 0) vars_0.load_wgt.write(data);
    else if (index == 1) vars_1.load_wgt.write(data);
    else if (index == 2) vars_2.load_wgt.write(data);
    else if (index == 3) vars_3.load_wgt.write(data);
    else if (index == 4) vars_4.load_wgt.write(data);
    else if (index == 5) vars_5.load_wgt.write(data);
    else if (index == 6) vars_6.load_wgt.write(data);
    else if (index == 7) vars_7.load_wgt.write(data);
    else vars_0.load_wgt.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) vars_0.wgt_fifo.write(data);
    else if (index == 1) vars_1.wgt_fifo.write(data);
    else if (index == 2) vars_2.wgt_fifo.write(data);
    else if (index == 3) vars_3.wgt_fifo.write(data);
    else if (index == 4) vars_4.wgt_fifo.write(data);
    else if (index == 5) vars_5.wgt_fifo.write(data);
    else if (index == 6) vars_6.wgt_fifo.write(data);
    else if (index == 7) vars_7.wgt_fifo.write(data);
    else vars_0.wgt_fifo.write(data);
  }

  void load_wsum_write(bool data, int index) {
    if (index == 0) vars_0.load_wsum.write(data);
    else if (index == 1) vars_1.load_wsum.write(data);
    else if (index == 2) vars_2.load_wsum.write(data);
    else if (index == 3) vars_3.load_wsum.write(data);
    else if (index == 4) vars_4.load_wsum.write(data);
    else if (index == 5) vars_5.load_wsum.write(data);
    else if (index == 6) vars_6.load_wsum.write(data);
    else if (index == 7) vars_7.load_wsum.write(data);
    else vars_0.load_wsum.write(data);
  }

  void wsum_write(bUF data, int index) {
    if (index == 0) vars_0.wsum_fifo.write(data);
    else if (index == 1) vars_1.wsum_fifo.write(data);
    else if (index == 2) vars_2.wsum_fifo.write(data);
    else if (index == 3) vars_3.wsum_fifo.write(data);
    else if (index == 4) vars_4.wsum_fifo.write(data);
    else if (index == 5) vars_5.wsum_fifo.write(data);
    else if (index == 6) vars_6.wsum_fifo.write(data);
    else if (index == 7) vars_7.wsum_fifo.write(data);
    else vars_0.wsum_fifo.write(data);
  }

  void crf_write(bUF data, int index) {
    if (index == 0) vars_0.crf_fifo.write(data);
    else if (index == 1) vars_1.crf_fifo.write(data);
    else if (index == 2) vars_2.crf_fifo.write(data);
    else if (index == 3) vars_3.crf_fifo.write(data);
    else if (index == 4) vars_4.crf_fifo.write(data);
    else if (index == 5) vars_5.crf_fifo.write(data);
    else if (index == 6) vars_6.crf_fifo.write(data);
    else if (index == 7) vars_7.crf_fifo.write(data);
    else vars_0.crf_fifo.write(data);
  }

  void crx_write(int data, int index) {
    if (index == 0) vars_0.crx_fifo.write(data);
    else if (index == 1) vars_1.crx_fifo.write(data);
    else if (index == 2) vars_2.crx_fifo.write(data);
    else if (index == 3) vars_3.crx_fifo.write(data);
    else if (index == 4) vars_4.crx_fifo.write(data);
    else if (index == 5) vars_5.crx_fifo.write(data);
    else if (index == 6) vars_6.crx_fifo.write(data);
    else if (index == 7) vars_7.crx_fifo.write(data);
    else vars_0.crx_fifo.write(data);
  }

  DATA dout_read(int index, int dout_index) {
    DATA d = {0, 0};
    if (index == 0 && dout_index == 0) return vars_0.dout1.read();
    else if (index == 0 && dout_index == 1) return vars_0.dout2.read();
    else if (index == 0 && dout_index == 2) return vars_0.dout3.read();
    else if (index == 0 && dout_index == 3) return vars_0.dout4.read();
    else if (index == 1 && dout_index == 0) return vars_1.dout1.read();
    else if (index == 1 && dout_index == 1) return vars_1.dout2.read();
    else if (index == 1 && dout_index == 2) return vars_1.dout3.read();
    else if (index == 1 && dout_index == 3) return vars_1.dout4.read();
    else if (index == 2 && dout_index == 0) return vars_2.dout1.read();
    else if (index == 2 && dout_index == 1) return vars_2.dout2.read();
    else if (index == 2 && dout_index == 2) return vars_2.dout3.read();
    else if (index == 2 && dout_index == 3) return vars_2.dout4.read();
    else if (index == 3 && dout_index == 0) return vars_3.dout1.read();
    else if (index == 3 && dout_index == 1) return vars_3.dout2.read();
    else if (index == 3 && dout_index == 2) return vars_3.dout3.read();
    else if (index == 3 && dout_index == 3) return vars_3.dout4.read();
    else if (index == 4 && dout_index == 0) return vars_4.dout1.read();
    else if (index == 4 && dout_index == 1) return vars_4.dout2.read();
    else if (index == 4 && dout_index == 2) return vars_4.dout3.read();
    else if (index == 4 && dout_index == 3) return vars_4.dout4.read();
    else if (index == 5 && dout_index == 0) return vars_5.dout1.read();
    else if (index == 5 && dout_index == 1) return vars_5.dout2.read();
    else if (index == 5 && dout_index == 2) return vars_5.dout3.read();
    else if (index == 5 && dout_index == 3) return vars_5.dout4.read();
    else if (index == 6 && dout_index == 0) return vars_6.dout1.read();
    else if (index == 6 && dout_index == 1) return vars_6.dout2.read();
    else if (index == 6 && dout_index == 2) return vars_6.dout3.read();
    else if (index == 6 && dout_index == 3) return vars_6.dout4.read();
    else if (index == 7 && dout_index == 0) return vars_7.dout1.read();
    else if (index == 7 && dout_index == 1) return vars_7.dout2.read();
    else if (index == 7 && dout_index == 2) return vars_7.dout3.read();
    else if (index == 7 && dout_index == 3) return vars_7.dout4.read();
    else return d;
  }

  bool check_ready(int index) {
    if (index == 0) return vars_0.ready.read();
    else if (index == 1) return vars_1.ready.read();
    else if (index == 2) return vars_2.ready.read();
    else if (index == 3) return vars_3.ready.read();
    else if (index == 4) return vars_4.ready.read();
    else if (index == 5) return vars_5.ready.read();
    else if (index == 6) return vars_6.ready.read();
    else if (index == 7) return vars_7.ready.read();
    else return vars_0.ready.read();
  }

  bool check_vmm_ready(int index) {
    if (index == 0) return vars_0.vmm_ready.read();
    else if (index == 1) return vars_1.vmm_ready.read();
    else if (index == 2) return vars_2.vmm_ready.read();
    else if (index == 3) return vars_3.vmm_ready.read();
    else if (index == 4) return vars_4.vmm_ready.read();
    else if (index == 5) return vars_5.vmm_ready.read();
    else if (index == 6) return vars_6.vmm_ready.read();
    else if (index == 7) return vars_7.vmm_ready.read();
    else return vars_0.vmm_ready.read();
  }

  bool check_post_ready(int index) {
    if (index == 0) return vars_0.post_ready.read();
    else if (index == 1) return vars_1.post_ready.read();
    else if (index == 2) return vars_2.post_ready.read();
    else if (index == 3) return vars_3.post_ready.read();
    else if (index == 4) return vars_4.post_ready.read();
    else if (index == 5) return vars_5.post_ready.read();
    else if (index == 6) return vars_6.post_ready.read();
    else if (index == 7) return vars_7.post_ready.read();
    else return vars_0.post_ready.read();
  }

  bool check_ppu_done(int index) {
    if (index == 0) return vars_0.ppu_done.read();
    else if (index == 1) return vars_1.ppu_done.read();
    else if (index == 2) return vars_2.ppu_done.read();
    else if (index == 3) return vars_3.ppu_done.read();
    else if (index == 4) return vars_4.ppu_done.read();
    else if (index == 5) return vars_5.ppu_done.read();
    else if (index == 6) return vars_6.ppu_done.read();
    else if (index == 7) return vars_7.ppu_done.read();
    else return vars_0.ppu_done.read();
  }

  void start_compute(int index, unsigned int w_idx, unsigned int wsum_idx,
                     unsigned int depth, int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    } else if (index == 1) {
      vars_1.ra.write(ra);
      vars_1.depth.write(depth);
      vars_1.w_idx.write(w_idx);
      vars_1.wsum_idx.write(wsum_idx);
      vars_1.compute.write(true);
    } else if (index == 2) {
      vars_2.ra.write(ra);
      vars_2.depth.write(depth);
      vars_2.w_idx.write(w_idx);
      vars_2.wsum_idx.write(wsum_idx);
      vars_2.compute.write(true);
    } else if (index == 3) {
      vars_3.ra.write(ra);
      vars_3.depth.write(depth);
      vars_3.w_idx.write(w_idx);
      vars_3.wsum_idx.write(wsum_idx);
      vars_3.compute.write(true);
    } else if (index == 4) {
      vars_4.ra.write(ra);
      vars_4.depth.write(depth);
      vars_4.w_idx.write(w_idx);
      vars_4.wsum_idx.write(wsum_idx);
      vars_4.compute.write(true);
    } else if (index == 5) {
      vars_5.ra.write(ra);
      vars_5.depth.write(depth);
      vars_5.w_idx.write(w_idx);
      vars_5.wsum_idx.write(wsum_idx);
      vars_5.compute.write(true);
    } else if (index == 6) {
      vars_6.ra.write(ra);
      vars_6.depth.write(depth);
      vars_6.w_idx.write(w_idx);
      vars_6.wsum_idx.write(wsum_idx);
      vars_6.compute.write(true);
    } else if (index == 7) {
      vars_7.ra.write(ra);
      vars_7.depth.write(depth);
      vars_7.w_idx.write(w_idx);
      vars_7.wsum_idx.write(wsum_idx);
      vars_7.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    }
  }

  void set_compute(int index, bool compute) {
    if (index == 0) vars_0.compute.write(compute);
    else if (index == 1) vars_1.compute.write(compute);
    else if (index == 2) vars_2.compute.write(compute);
    else if (index == 3) vars_3.compute.write(compute);
    else if (index == 4) vars_4.compute.write(compute);
    else if (index == 5) vars_5.compute.write(compute);
    else if (index == 6) vars_6.compute.write(compute);
    else if (index == 7) vars_7.compute.write(compute);
    else vars_0.compute.write(compute);
  }

  VMM_vars &operator[](int index) {
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

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    V0.init(clock, reset, vars_0);
    V1.init(clock, reset, vars_1);
    V2.init(clock, reset, vars_2);
    V3.init(clock, reset, vars_3);
    V4.init(clock, reset, vars_4);
    V5.init(clock, reset, vars_5);
    V6.init(clock, reset, vars_6);
    V7.init(clock, reset, vars_7);
  }
};

struct var_array16 {
  VMM_vars vars_0;
  VMM_vars vars_1;
  VMM_vars vars_2;
  VMM_vars vars_3;
  VMM_vars vars_4;
  VMM_vars vars_5;
  VMM_vars vars_6;
  VMM_vars vars_7;
  VMM_vars vars_8;
  VMM_vars vars_9;
  VMM_vars vars_10;
  VMM_vars vars_11;
  VMM_vars vars_12;
  VMM_vars vars_13;
  VMM_vars vars_14;
  VMM_vars vars_15;
  VMM_UNIT V0;
  VMM_UNIT V1;
  VMM_UNIT V2;
  VMM_UNIT V3;
  VMM_UNIT V4;
  VMM_UNIT V5;
  VMM_UNIT V6;
  VMM_UNIT V7;
  VMM_UNIT V8;
  VMM_UNIT V9;
  VMM_UNIT V10;
  VMM_UNIT V11;
  VMM_UNIT V12;
  VMM_UNIT V13;
  VMM_UNIT V14;
  VMM_UNIT V15;

#ifndef __SYNTHESIS__
  var_array16()
      : vars_0(16, 0), vars_1(16, 1), vars_2(16, 2), vars_3(16, 3),
        vars_4(16, 4), vars_5(16, 5), vars_6(16, 6), vars_7(16, 7),
        vars_8(16, 8), vars_9(16, 9), vars_10(16, 10), vars_11(16, 11),
        vars_12(16, 12), vars_13(16, 13), vars_14(16, 14), vars_15(16, 15),
        V0("V0"), V1("V1"), V2("V2"), V3("V3"), V4("V4"), V5("V5"), V6("V6"),
        V7("V7"), V8("V8"), V9("V9"), V10("V10"), V11("V11"), V12("V12"),
        V13("V13"), V14("V14"), V15("V15") {}
#else
  var_array16()
      : vars_0(16), vars_1(16), vars_2(16), vars_3(16), vars_4(16), vars_5(16),
        vars_6(16), vars_7(16), vars_8(16), vars_9(16), vars_10(16),
        vars_11(16), vars_12(16), vars_13(16), vars_14(16), vars_15(16),
        V0("V0"), V1("V1"), V2("V2"), V3("V3"), V4("V4"), V5("V5"), V6("V6"),
        V7("V7"), V8("V8"), V9("V9"), V10("V10"), V11("V11"), V12("V12"),
        V13("V13"), V14("V14"), V15("V15") {}
#endif

  bool check_douts_empty(int index) {
    bool empty = true;
    if (index == 0)
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    else if (index == 1)
      empty = !(vars_1.dout1.num_available() || vars_1.dout2.num_available() ||
                vars_1.dout3.num_available() || vars_1.dout4.num_available());
    else if (index == 2)
      empty = !(vars_2.dout1.num_available() || vars_2.dout2.num_available() ||
                vars_2.dout3.num_available() || vars_2.dout4.num_available());
    else if (index == 3)
      empty = !(vars_3.dout1.num_available() || vars_3.dout2.num_available() ||
                vars_3.dout3.num_available() || vars_3.dout4.num_available());
    else if (index == 4)
      empty = !(vars_4.dout1.num_available() || vars_4.dout2.num_available() ||
                vars_4.dout3.num_available() || vars_4.dout4.num_available());
    else if (index == 5)
      empty = !(vars_5.dout1.num_available() || vars_5.dout2.num_available() ||
                vars_5.dout3.num_available() || vars_5.dout4.num_available());
    else if (index == 6)
      empty = !(vars_6.dout1.num_available() || vars_6.dout2.num_available() ||
                vars_6.dout3.num_available() || vars_6.dout4.num_available());
    else if (index == 7)
      empty = !(vars_7.dout1.num_available() || vars_7.dout2.num_available() ||
                vars_7.dout3.num_available() || vars_7.dout4.num_available());
    else if (index == 8)
      empty = !(vars_8.dout1.num_available() || vars_8.dout2.num_available() ||
                vars_8.dout3.num_available() || vars_8.dout4.num_available());
    else if (index == 9)
      empty = !(vars_9.dout1.num_available() || vars_9.dout2.num_available() ||
                vars_9.dout3.num_available() || vars_9.dout4.num_available());
    else if (index == 10)
      empty =
          !(vars_10.dout1.num_available() || vars_10.dout2.num_available() ||
            vars_10.dout3.num_available() || vars_10.dout4.num_available());
    else if (index == 11)
      empty =
          !(vars_11.dout1.num_available() || vars_11.dout2.num_available() ||
            vars_11.dout3.num_available() || vars_11.dout4.num_available());
    else if (index == 12)
      empty =
          !(vars_12.dout1.num_available() || vars_12.dout2.num_available() ||
            vars_12.dout3.num_available() || vars_12.dout4.num_available());
    else if (index == 13)
      empty =
          !(vars_13.dout1.num_available() || vars_13.dout2.num_available() ||
            vars_13.dout3.num_available() || vars_13.dout4.num_available());
    else if (index == 14)
      empty =
          !(vars_14.dout1.num_available() || vars_14.dout2.num_available() ||
            vars_14.dout3.num_available() || vars_14.dout4.num_available());
    else if (index == 15)
      empty =
          !(vars_15.dout1.num_available() || vars_15.dout2.num_available() ||
            vars_15.dout3.num_available() || vars_15.dout4.num_available());
    else
      empty = !(vars_0.dout1.num_available() || vars_0.dout2.num_available() ||
                vars_0.dout3.num_available() || vars_0.dout4.num_available());
    return empty;
  }

  int next(int index) {
    if (index == 0) return 1;
    else if (index == 1) return 2;
    else if (index == 2) return 3;
    else if (index == 3) return 4;
    else if (index == 4) return 5;
    else if (index == 5) return 6;
    else if (index == 6) return 7;
    else if (index == 7) return 8;
    else if (index == 8) return 9;
    else if (index == 9) return 10;
    else if (index == 10) return 11;
    else if (index == 11) return 12;
    else if (index == 12) return 13;
    else if (index == 13) return 14;
    else if (index == 14) return 15;
    else if (index == 15) return 0;
    else return 0;
  }

  void post_write(int data, int index) {
    if (index == 0) vars_0.post_fifo.write(data);
    else if (index == 1) vars_1.post_fifo.write(data);
    else if (index == 2) vars_2.post_fifo.write(data);
    else if (index == 3) vars_3.post_fifo.write(data);
    else if (index == 4) vars_4.post_fifo.write(data);
    else if (index == 5) vars_5.post_fifo.write(data);
    else if (index == 6) vars_6.post_fifo.write(data);
    else if (index == 7) vars_7.post_fifo.write(data);
    else if (index == 8) vars_8.post_fifo.write(data);
    else if (index == 9) vars_9.post_fifo.write(data);
    else if (index == 10) vars_10.post_fifo.write(data);
    else if (index == 11) vars_11.post_fifo.write(data);
    else if (index == 12) vars_12.post_fifo.write(data);
    else if (index == 13) vars_13.post_fifo.write(data);
    else if (index == 14) vars_14.post_fifo.write(data);
    else if (index == 15) vars_15.post_fifo.write(data);
    else vars_0.post_fifo.write(data);
  }

  void send_done_write(bool data, int index) {
    if (index == 0) vars_0.send_done.write(data);
    else if (index == 1) vars_1.send_done.write(data);
    else if (index == 2) vars_2.send_done.write(data);
    else if (index == 3) vars_3.send_done.write(data);
    else if (index == 4) vars_4.send_done.write(data);
    else if (index == 5) vars_5.send_done.write(data);
    else if (index == 6) vars_6.send_done.write(data);
    else if (index == 7) vars_7.send_done.write(data);
    else if (index == 8) vars_8.send_done.write(data);
    else if (index == 9) vars_9.send_done.write(data);
    else if (index == 10) vars_10.send_done.write(data);
    else if (index == 11) vars_11.send_done.write(data);
    else if (index == 12) vars_12.send_done.write(data);
    else if (index == 13) vars_13.send_done.write(data);
    else if (index == 14) vars_14.send_done.write(data);
    else if (index == 15) vars_15.send_done.write(data);
    else vars_0.send_done.write(data);
  }

  void load_inp_write(bool data, int index) {
    if (index == 0) vars_0.load_inp.write(data);
    else if (index == 1) vars_1.load_inp.write(data);
    else if (index == 2) vars_2.load_inp.write(data);
    else if (index == 3) vars_3.load_inp.write(data);
    else if (index == 4) vars_4.load_inp.write(data);
    else if (index == 5) vars_5.load_inp.write(data);
    else if (index == 6) vars_6.load_inp.write(data);
    else if (index == 7) vars_7.load_inp.write(data);
    else if (index == 8) vars_8.load_inp.write(data);
    else if (index == 9) vars_9.load_inp.write(data);
    else if (index == 10) vars_10.load_inp.write(data);
    else if (index == 11) vars_11.load_inp.write(data);
    else if (index == 12) vars_12.load_inp.write(data);
    else if (index == 13) vars_13.load_inp.write(data);
    else if (index == 14) vars_14.load_inp.write(data);
    else if (index == 15) vars_15.load_inp.write(data);
    else vars_0.load_inp.write(data);
  }

  void inp_len_write(unsigned int len, int index) {
    if (index == 0) vars_0.inp_len.write(len);
    else if (index == 1) vars_1.inp_len.write(len);
    else if (index == 2) vars_2.inp_len.write(len);
    else if (index == 3) vars_3.inp_len.write(len);
    else if (index == 4) vars_4.inp_len.write(len);
    else if (index == 5) vars_5.inp_len.write(len);
    else if (index == 6) vars_6.inp_len.write(len);
    else if (index == 7) vars_7.inp_len.write(len);
    else if (index == 8) vars_8.inp_len.write(len);
    else if (index == 9) vars_9.inp_len.write(len);
    else if (index == 10) vars_10.inp_len.write(len);
    else if (index == 11) vars_11.inp_len.write(len);
    else if (index == 12) vars_12.inp_len.write(len);
    else if (index == 13) vars_13.inp_len.write(len);
    else if (index == 14) vars_14.inp_len.write(len);
    else if (index == 15) vars_15.inp_len.write(len);
    else vars_0.inp_len.write(len);
  }

  void inp_write(bUF data, int index) {
    // if (index == 0) vars_0.inp_fifo.write(data);
    // else if (index == 1) vars_1.inp_fifo.write(data);
    // else if (index == 2) vars_2.inp_fifo.write(data);
    // else if (index == 3) vars_3.inp_fifo.write(data);
    // else if (index == 4) vars_4.inp_fifo.write(data);
    // else if (index == 5) vars_5.inp_fifo.write(data);
    // else if (index == 6) vars_6.inp_fifo.write(data);
    // else if (index == 7) vars_7.inp_fifo.write(data);
    // else vars_0.inp_fifo.write(data);
    vars_0.inp_fifo.write(data);
    vars_1.inp_fifo.write(data);
    vars_2.inp_fifo.write(data);
    vars_3.inp_fifo.write(data);
    vars_4.inp_fifo.write(data);
    vars_5.inp_fifo.write(data);
    vars_6.inp_fifo.write(data);
    vars_7.inp_fifo.write(data);
    vars_8.inp_fifo.write(data);
    vars_9.inp_fifo.write(data);
    vars_10.inp_fifo.write(data);
    vars_11.inp_fifo.write(data);
    vars_12.inp_fifo.write(data);
    vars_13.inp_fifo.write(data);
    vars_14.inp_fifo.write(data);
    vars_15.inp_fifo.write(data);
  }

  void load_wgt_write(bool data, int index) {
    if (index == 0) vars_0.load_wgt.write(data);
    else if (index == 1) vars_1.load_wgt.write(data);
    else if (index == 2) vars_2.load_wgt.write(data);
    else if (index == 3) vars_3.load_wgt.write(data);
    else if (index == 4) vars_4.load_wgt.write(data);
    else if (index == 5) vars_5.load_wgt.write(data);
    else if (index == 6) vars_6.load_wgt.write(data);
    else if (index == 7) vars_7.load_wgt.write(data);
    else if (index == 8) vars_8.load_wgt.write(data);
    else if (index == 9) vars_9.load_wgt.write(data);
    else if (index == 10) vars_10.load_wgt.write(data);
    else if (index == 11) vars_11.load_wgt.write(data);
    else if (index == 12) vars_12.load_wgt.write(data);
    else if (index == 13) vars_13.load_wgt.write(data);
    else if (index == 14) vars_14.load_wgt.write(data);
    else if (index == 15) vars_15.load_wgt.write(data);
    else vars_0.load_wgt.write(data);
  }

  void wgt_write(bUF data, int index) {
    if (index == 0) vars_0.wgt_fifo.write(data);
    else if (index == 1) vars_1.wgt_fifo.write(data);
    else if (index == 2) vars_2.wgt_fifo.write(data);
    else if (index == 3) vars_3.wgt_fifo.write(data);
    else if (index == 4) vars_4.wgt_fifo.write(data);
    else if (index == 5) vars_5.wgt_fifo.write(data);
    else if (index == 6) vars_6.wgt_fifo.write(data);
    else if (index == 7) vars_7.wgt_fifo.write(data);
    else if (index == 8) vars_8.wgt_fifo.write(data);
    else if (index == 9) vars_9.wgt_fifo.write(data);
    else if (index == 10) vars_10.wgt_fifo.write(data);
    else if (index == 11) vars_11.wgt_fifo.write(data);
    else if (index == 12) vars_12.wgt_fifo.write(data);
    else if (index == 13) vars_13.wgt_fifo.write(data);
    else if (index == 14) vars_14.wgt_fifo.write(data);
    else if (index == 15) vars_15.wgt_fifo.write(data);
    else vars_0.wgt_fifo.write(data);
  }

  void load_wsum_write(bool data, int index) {
    if (index == 0) vars_0.load_wsum.write(data);
    else if (index == 1) vars_1.load_wsum.write(data);
    else if (index == 2) vars_2.load_wsum.write(data);
    else if (index == 3) vars_3.load_wsum.write(data);
    else if (index == 4) vars_4.load_wsum.write(data);
    else if (index == 5) vars_5.load_wsum.write(data);
    else if (index == 6) vars_6.load_wsum.write(data);
    else if (index == 7) vars_7.load_wsum.write(data);
    else if (index == 8) vars_8.load_wsum.write(data);
    else if (index == 9) vars_9.load_wsum.write(data);
    else if (index == 10) vars_10.load_wsum.write(data);
    else if (index == 11) vars_11.load_wsum.write(data);
    else if (index == 12) vars_12.load_wsum.write(data);
    else if (index == 13) vars_13.load_wsum.write(data);
    else if (index == 14) vars_14.load_wsum.write(data);
    else if (index == 15) vars_15.load_wsum.write(data);
    else vars_0.load_wsum.write(data);
  }

  void wsum_write(bUF data, int index) {
    if (index == 0) vars_0.wsum_fifo.write(data);
    else if (index == 1) vars_1.wsum_fifo.write(data);
    else if (index == 2) vars_2.wsum_fifo.write(data);
    else if (index == 3) vars_3.wsum_fifo.write(data);
    else if (index == 4) vars_4.wsum_fifo.write(data);
    else if (index == 5) vars_5.wsum_fifo.write(data);
    else if (index == 6) vars_6.wsum_fifo.write(data);
    else if (index == 7) vars_7.wsum_fifo.write(data);
    else if (index == 8) vars_8.wsum_fifo.write(data);
    else if (index == 9) vars_9.wsum_fifo.write(data);
    else if (index == 10) vars_10.wsum_fifo.write(data);
    else if (index == 11) vars_11.wsum_fifo.write(data);
    else if (index == 12) vars_12.wsum_fifo.write(data);
    else if (index == 13) vars_13.wsum_fifo.write(data);
    else if (index == 14) vars_14.wsum_fifo.write(data);
    else if (index == 15) vars_15.wsum_fifo.write(data);
    else vars_0.wsum_fifo.write(data);
  }

  void crf_write(bUF data, int index) {
    if (index == 0) vars_0.crf_fifo.write(data);
    else if (index == 1) vars_1.crf_fifo.write(data);
    else if (index == 2) vars_2.crf_fifo.write(data);
    else if (index == 3) vars_3.crf_fifo.write(data);
    else if (index == 4) vars_4.crf_fifo.write(data);
    else if (index == 5) vars_5.crf_fifo.write(data);
    else if (index == 6) vars_6.crf_fifo.write(data);
    else if (index == 7) vars_7.crf_fifo.write(data);
    else if (index == 8) vars_8.crf_fifo.write(data);
    else if (index == 9) vars_9.crf_fifo.write(data);
    else if (index == 10) vars_10.crf_fifo.write(data);
    else if (index == 11) vars_11.crf_fifo.write(data);
    else if (index == 12) vars_12.crf_fifo.write(data);
    else if (index == 13) vars_13.crf_fifo.write(data);
    else if (index == 14) vars_14.crf_fifo.write(data);
    else if (index == 15) vars_15.crf_fifo.write(data);
    else vars_0.crf_fifo.write(data);
  }

  void crx_write(int data, int index) {
    if (index == 0) vars_0.crx_fifo.write(data);
    else if (index == 1) vars_1.crx_fifo.write(data);
    else if (index == 2) vars_2.crx_fifo.write(data);
    else if (index == 3) vars_3.crx_fifo.write(data);
    else if (index == 4) vars_4.crx_fifo.write(data);
    else if (index == 5) vars_5.crx_fifo.write(data);
    else if (index == 6) vars_6.crx_fifo.write(data);
    else if (index == 7) vars_7.crx_fifo.write(data);
    else if (index == 8) vars_8.crx_fifo.write(data);
    else if (index == 9) vars_9.crx_fifo.write(data);
    else if (index == 10) vars_10.crx_fifo.write(data);
    else if (index == 11) vars_11.crx_fifo.write(data);
    else if (index == 12) vars_12.crx_fifo.write(data);
    else if (index == 13) vars_13.crx_fifo.write(data);
    else if (index == 14) vars_14.crx_fifo.write(data);
    else if (index == 15) vars_15.crx_fifo.write(data);
    else vars_0.crx_fifo.write(data);
  }

  DATA dout_read(int index, int dout_index) {
    DATA d = {0, 0};

    if (index == 0 && dout_index == 0) return vars_0.dout1.read();
    else if (index == 0 && dout_index == 1) return vars_0.dout2.read();
    else if (index == 0 && dout_index == 2) return vars_0.dout3.read();
    else if (index == 0 && dout_index == 3) return vars_0.dout4.read();
    else if (index == 1 && dout_index == 0) return vars_1.dout1.read();
    else if (index == 1 && dout_index == 1) return vars_1.dout2.read();
    else if (index == 1 && dout_index == 2) return vars_1.dout3.read();
    else if (index == 1 && dout_index == 3) return vars_1.dout4.read();
    else if (index == 2 && dout_index == 0) return vars_2.dout1.read();
    else if (index == 2 && dout_index == 1) return vars_2.dout2.read();
    else if (index == 2 && dout_index == 2) return vars_2.dout3.read();
    else if (index == 2 && dout_index == 3) return vars_2.dout4.read();
    else if (index == 3 && dout_index == 0) return vars_3.dout1.read();
    else if (index == 3 && dout_index == 1) return vars_3.dout2.read();
    else if (index == 3 && dout_index == 2) return vars_3.dout3.read();
    else if (index == 3 && dout_index == 3) return vars_3.dout4.read();
    else if (index == 4 && dout_index == 0) return vars_4.dout1.read();
    else if (index == 4 && dout_index == 1) return vars_4.dout2.read();
    else if (index == 4 && dout_index == 2) return vars_4.dout3.read();
    else if (index == 4 && dout_index == 3) return vars_4.dout4.read();
    else if (index == 5 && dout_index == 0) return vars_5.dout1.read();
    else if (index == 5 && dout_index == 1) return vars_5.dout2.read();
    else if (index == 5 && dout_index == 2) return vars_5.dout3.read();
    else if (index == 5 && dout_index == 3) return vars_5.dout4.read();
    else if (index == 6 && dout_index == 0) return vars_6.dout1.read();
    else if (index == 6 && dout_index == 1) return vars_6.dout2.read();
    else if (index == 6 && dout_index == 2) return vars_6.dout3.read();
    else if (index == 6 && dout_index == 3) return vars_6.dout4.read();
    else if (index == 7 && dout_index == 0) return vars_7.dout1.read();
    else if (index == 7 && dout_index == 1) return vars_7.dout2.read();
    else if (index == 7 && dout_index == 2) return vars_7.dout3.read();
    else if (index == 7 && dout_index == 3) return vars_7.dout4.read();
    else if (index == 8 && dout_index == 0) return vars_8.dout1.read();
    else if (index == 8 && dout_index == 1) return vars_8.dout2.read();
    else if (index == 8 && dout_index == 2) return vars_8.dout3.read();
    else if (index == 8 && dout_index == 3) return vars_8.dout4.read();
    else if (index == 9 && dout_index == 0) return vars_9.dout1.read();
    else if (index == 9 && dout_index == 1) return vars_9.dout2.read();
    else if (index == 9 && dout_index == 2) return vars_9.dout3.read();
    else if (index == 9 && dout_index == 3) return vars_9.dout4.read();
    else if (index == 10 && dout_index == 0) return vars_10.dout1.read();
    else if (index == 10 && dout_index == 1) return vars_10.dout2.read();
    else if (index == 10 && dout_index == 2) return vars_10.dout3.read();
    else if (index == 10 && dout_index == 3) return vars_10.dout4.read();
    else if (index == 11 && dout_index == 0) return vars_11.dout1.read();
    else if (index == 11 && dout_index == 1) return vars_11.dout2.read();
    else if (index == 11 && dout_index == 2) return vars_11.dout3.read();
    else if (index == 11 && dout_index == 3) return vars_11.dout4.read();
    else if (index == 12 && dout_index == 0) return vars_12.dout1.read();
    else if (index == 12 && dout_index == 1) return vars_12.dout2.read();
    else if (index == 12 && dout_index == 2) return vars_12.dout3.read();
    else if (index == 12 && dout_index == 3) return vars_12.dout4.read();
    else if (index == 13 && dout_index == 0) return vars_13.dout1.read();
    else if (index == 13 && dout_index == 1) return vars_13.dout2.read();
    else if (index == 13 && dout_index == 2) return vars_13.dout3.read();
    else if (index == 13 && dout_index == 3) return vars_13.dout4.read();
    else if (index == 14 && dout_index == 0) return vars_14.dout1.read();
    else if (index == 14 && dout_index == 1) return vars_14.dout2.read();
    else if (index == 14 && dout_index == 2) return vars_14.dout3.read();
    else if (index == 14 && dout_index == 3) return vars_14.dout4.read();
    else if (index == 15 && dout_index == 0) return vars_15.dout1.read();
    else if (index == 15 && dout_index == 1) return vars_15.dout2.read();
    else if (index == 15 && dout_index == 2) return vars_15.dout3.read();
    else if (index == 15 && dout_index == 3) return vars_15.dout4.read();
    else return d;
  }

  bool check_ready(int index) {
    if (index == 0) return vars_0.ready.read();
    else if (index == 1) return vars_1.ready.read();
    else if (index == 2) return vars_2.ready.read();
    else if (index == 3) return vars_3.ready.read();
    else if (index == 4) return vars_4.ready.read();
    else if (index == 5) return vars_5.ready.read();
    else if (index == 6) return vars_6.ready.read();
    else if (index == 7) return vars_7.ready.read();
    else if (index == 8) return vars_8.ready.read();
    else if (index == 9) return vars_9.ready.read();
    else if (index == 10) return vars_10.ready.read();
    else if (index == 11) return vars_11.ready.read();
    else if (index == 12) return vars_12.ready.read();
    else if (index == 13) return vars_13.ready.read();
    else if (index == 14) return vars_14.ready.read();
    else if (index == 15) return vars_15.ready.read();
    else return vars_0.ready.read();
  }

  bool check_vmm_ready(int index) {
    if (index == 0) return vars_0.vmm_ready.read();
    else if (index == 1) return vars_1.vmm_ready.read();
    else if (index == 2) return vars_2.vmm_ready.read();
    else if (index == 3) return vars_3.vmm_ready.read();
    else if (index == 4) return vars_4.vmm_ready.read();
    else if (index == 5) return vars_5.vmm_ready.read();
    else if (index == 6) return vars_6.vmm_ready.read();
    else if (index == 7) return vars_7.vmm_ready.read();
    else if (index == 8) return vars_8.vmm_ready.read();
    else if (index == 9) return vars_9.vmm_ready.read();
    else if (index == 10) return vars_10.vmm_ready.read();
    else if (index == 11) return vars_11.vmm_ready.read();
    else if (index == 12) return vars_12.vmm_ready.read();
    else if (index == 13) return vars_13.vmm_ready.read();
    else if (index == 14) return vars_14.vmm_ready.read();
    else if (index == 15) return vars_15.vmm_ready.read();
    else return vars_0.vmm_ready.read();
  }

  bool check_post_ready(int index) {
    if (index == 0) return vars_0.post_ready.read();
    else if (index == 1) return vars_1.post_ready.read();
    else if (index == 2) return vars_2.post_ready.read();
    else if (index == 3) return vars_3.post_ready.read();
    else if (index == 4) return vars_4.post_ready.read();
    else if (index == 5) return vars_5.post_ready.read();
    else if (index == 6) return vars_6.post_ready.read();
    else if (index == 7) return vars_7.post_ready.read();
    else if (index == 8) return vars_8.post_ready.read();
    else if (index == 9) return vars_9.post_ready.read();
    else if (index == 10) return vars_10.post_ready.read();
    else if (index == 11) return vars_11.post_ready.read();
    else if (index == 12) return vars_12.post_ready.read();
    else if (index == 13) return vars_13.post_ready.read();
    else if (index == 14) return vars_14.post_ready.read();
    else if (index == 15) return vars_15.post_ready.read();
    else return vars_0.post_ready.read();
  }

  bool check_ppu_done(int index) {
    if (index == 0) return vars_0.ppu_done.read();
    else if (index == 1) return vars_1.ppu_done.read();
    else if (index == 2) return vars_2.ppu_done.read();
    else if (index == 3) return vars_3.ppu_done.read();
    else if (index == 4) return vars_4.ppu_done.read();
    else if (index == 5) return vars_5.ppu_done.read();
    else if (index == 6) return vars_6.ppu_done.read();
    else if (index == 7) return vars_7.ppu_done.read();
    else if (index == 8) return vars_8.ppu_done.read();
    else if (index == 9) return vars_9.ppu_done.read();
    else if (index == 10) return vars_10.ppu_done.read();
    else if (index == 11) return vars_11.ppu_done.read();
    else if (index == 12) return vars_12.ppu_done.read();
    else if (index == 13) return vars_13.ppu_done.read();
    else if (index == 14) return vars_14.ppu_done.read();
    else if (index == 15) return vars_15.ppu_done.read();
    else return vars_0.ppu_done.read();
  }

  void start_compute(int index, unsigned int w_idx, unsigned int wsum_idx,
                     unsigned int depth, int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    } else if (index == 1) {
      vars_1.ra.write(ra);
      vars_1.depth.write(depth);
      vars_1.w_idx.write(w_idx);
      vars_1.wsum_idx.write(wsum_idx);
      vars_1.compute.write(true);
    } else if (index == 2) {
      vars_2.ra.write(ra);
      vars_2.depth.write(depth);
      vars_2.w_idx.write(w_idx);
      vars_2.wsum_idx.write(wsum_idx);
      vars_2.compute.write(true);
    } else if (index == 3) {
      vars_3.ra.write(ra);
      vars_3.depth.write(depth);
      vars_3.w_idx.write(w_idx);
      vars_3.wsum_idx.write(wsum_idx);
      vars_3.compute.write(true);
    } else if (index == 4) {
      vars_4.ra.write(ra);
      vars_4.depth.write(depth);
      vars_4.w_idx.write(w_idx);
      vars_4.wsum_idx.write(wsum_idx);
      vars_4.compute.write(true);
    } else if (index == 5) {
      vars_5.ra.write(ra);
      vars_5.depth.write(depth);
      vars_5.w_idx.write(w_idx);
      vars_5.wsum_idx.write(wsum_idx);
      vars_5.compute.write(true);
    } else if (index == 6) {
      vars_6.ra.write(ra);
      vars_6.depth.write(depth);
      vars_6.w_idx.write(w_idx);
      vars_6.wsum_idx.write(wsum_idx);
      vars_6.compute.write(true);
    } else if (index == 7) {
      vars_7.ra.write(ra);
      vars_7.depth.write(depth);
      vars_7.w_idx.write(w_idx);
      vars_7.wsum_idx.write(wsum_idx);
      vars_7.compute.write(true);
    } else if (index == 8) {
      vars_8.ra.write(ra);
      vars_8.depth.write(depth);
      vars_8.w_idx.write(w_idx);
      vars_8.wsum_idx.write(wsum_idx);
      vars_8.compute.write(true);
    } else if (index == 9) {
      vars_9.ra.write(ra);
      vars_9.depth.write(depth);
      vars_9.w_idx.write(w_idx);
      vars_9.wsum_idx.write(wsum_idx);
      vars_9.compute.write(true);
    } else if (index == 10) {
      vars_10.ra.write(ra);
      vars_10.depth.write(depth);
      vars_10.w_idx.write(w_idx);
      vars_10.wsum_idx.write(wsum_idx);
      vars_10.compute.write(true);
    } else if (index == 11) {
      vars_11.ra.write(ra);
      vars_11.depth.write(depth);
      vars_11.w_idx.write(w_idx);
      vars_11.wsum_idx.write(wsum_idx);
      vars_11.compute.write(true);
    } else if (index == 12) {
      vars_12.ra.write(ra);
      vars_12.depth.write(depth);
      vars_12.w_idx.write(w_idx);
      vars_12.wsum_idx.write(wsum_idx);
      vars_12.compute.write(true);
    } else if (index == 13) {
      vars_13.ra.write(ra);
      vars_13.depth.write(depth);
      vars_13.w_idx.write(w_idx);
      vars_13.wsum_idx.write(wsum_idx);
      vars_13.compute.write(true);
    } else if (index == 14) {
      vars_14.ra.write(ra);
      vars_14.depth.write(depth);
      vars_14.w_idx.write(w_idx);
      vars_14.wsum_idx.write(wsum_idx);
      vars_14.compute.write(true);
    } else if (index == 15) {
      vars_15.ra.write(ra);
      vars_15.depth.write(depth);
      vars_15.w_idx.write(w_idx);
      vars_15.wsum_idx.write(wsum_idx);
      vars_15.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.wsum_idx.write(wsum_idx);
      vars_0.compute.write(true);
    }
  }

  void set_compute(int index, bool compute) {
    if (index == 0) vars_0.compute.write(compute);
    else if (index == 1) vars_1.compute.write(compute);
    else if (index == 2) vars_2.compute.write(compute);
    else if (index == 3) vars_3.compute.write(compute);
    else if (index == 4) vars_4.compute.write(compute);
    else if (index == 5) vars_5.compute.write(compute);
    else if (index == 6) vars_6.compute.write(compute);
    else if (index == 7) vars_7.compute.write(compute);
    else if (index == 8) vars_8.compute.write(compute);
    else if (index == 9) vars_9.compute.write(compute);
    else if (index == 10) vars_10.compute.write(compute);
    else if (index == 11) vars_11.compute.write(compute);
    else if (index == 12) vars_12.compute.write(compute);
    else if (index == 13) vars_13.compute.write(compute);
    else if (index == 14) vars_14.compute.write(compute);
    else if (index == 15) vars_15.compute.write(compute);
    else vars_0.compute.write(compute);
  }

  VMM_vars &operator[](int index) {
    if (index == 0) return vars_0;
    else if (index == 1) return vars_1;
    else if (index == 2) return vars_2;
    else if (index == 3) return vars_3;
    else if (index == 4) return vars_4;
    else if (index == 5) return vars_5;
    else if (index == 6) return vars_6;
    else if (index == 7) return vars_7;
    else if (index == 8) return vars_8;
    else if (index == 9) return vars_9;
    else if (index == 10) return vars_10;
    else if (index == 11) return vars_11;
    else if (index == 12) return vars_12;
    else if (index == 13) return vars_13;
    else if (index == 14) return vars_14;
    else if (index == 15) return vars_15;
    else return vars_0;
  }

  void init(sc_in<bool> &clock, sc_in<bool> &reset) {
    V0.init(clock, reset, vars_0);
    V1.init(clock, reset, vars_1);
    V2.init(clock, reset, vars_2);
    V3.init(clock, reset, vars_3);
    V4.init(clock, reset, vars_4);
    V5.init(clock, reset, vars_5);
    V6.init(clock, reset, vars_6);
    V7.init(clock, reset, vars_7);
    V8.init(clock, reset, vars_8);
    V9.init(clock, reset, vars_9);
    V10.init(clock, reset, vars_10);
    V11.init(clock, reset, vars_11);
    V12.init(clock, reset, vars_12);
    V13.init(clock, reset, vars_13);
    V14.init(clock, reset, vars_14);
    V15.init(clock, reset, vars_15);
  }
};

#endif // VMM_UNIT_H
