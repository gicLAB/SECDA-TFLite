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
  sc_in<bool> compute;
  sc_in<bool> send_done;
  sc_out<bool> ready;
  sc_out<bool> vmm_ready;
  sc_out<bool> ppu_done;

  sc_in<int> ra;
  sc_in<unsigned int> depth;
  sc_in<unsigned int> w_idx;
  sc_in<unsigned int> wgt_len;
  sc_in<unsigned int> inp_len;

  // FIFOs
  sc_fifo_in<bUF> wgt_fifo;
  sc_fifo_in<bUF> inp_fifo;
  sc_fifo_in<int> post_fifo;

  sc_fifo_out<DATA> dout1;
  sc_fifo_out<DATA> dout2;
  sc_fifo_out<DATA> dout3;
  sc_fifo_out<DATA> dout4;

  // Signals
#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> post_ready;
#else
  sc_signal<bool> post_ready;
#endif

  // Memory
  ACC_DTYPE<32> inp_1a_1[INP_BUF_LEN];
  ACC_DTYPE<32> inp_1b_1[INP_BUF_LEN];
  ACC_DTYPE<32> inp_1c_1[INP_BUF_LEN];
  ACC_DTYPE<32> inp_1d_1[INP_BUF_LEN];
  
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data1a[WGT_BUF_LEN];
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data2a[WGT_BUF_LEN];
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data3a[WGT_BUF_LEN];
  ACC_DTYPE<WGT_BRAM_DATAWIDTH> wgt_data4a[WGT_BUF_LEN];
  ACC_DTYPE<32> out[16][4];
  ACC_DTYPE<32> g[16];
  ACC_DTYPE<8> r[16];

  // Debug
  // sc_out_sig computeS;
  sc_out<int> computeS;
  sc_out<int> postS;

  // functions
  // sc_int<PROD_DATA_WIDTH> mul_s8(sc_int<8>, sc_int<4>);
  sc_int<PROD_DATA_WIDTH> mul_s8(sc_int<4>, sc_int<8>);

  sc_int<64> mul_s64(int, sc_int<64>);

  void VM_PE(ACC_DTYPE<WGT_BRAM_DATAWIDTH> *, ACC_DTYPE<WGT_BRAM_DATAWIDTH> *, ACC_DTYPE<WGT_BRAM_DATAWIDTH> *, ACC_DTYPE<WGT_BRAM_DATAWIDTH> *,
             ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *, ACC_DTYPE<32> *,
             ACC_DTYPE<32>[][4], int, int, int);

  int Quantised_Multiplier_gemmlowp(int, int, sc_int<64>, sc_int<32>, sc_int<32>,
                              sc_int<32>);

  int Quantised_Multiplier_ruy_reference(int, int, sc_int<8>);

  void PPU(int *, int *, int *, sc_int<8> *, ACC_DTYPE<32> *, ACC_DTYPE<8> *);

  // modules
  void LoadWeights();

  void LoadInputs();

  void Compute();

  void Post();

  void init(sc_in<bool> & clock, sc_in<bool> & reset, VMM_vars & vars) {
    this->clock(clock);
    this->reset(reset);
    this->load_inp(vars.load_inp);
    this->load_wgt(vars.load_wgt);
    this->compute(vars.compute);
    this->send_done(vars.send_done);
    this->ready(vars.ready);
    this->vmm_ready(vars.vmm_ready);
    this->ppu_done(vars.ppu_done);
    this->ra(vars.ra);
    this->depth(vars.depth);
    this->w_idx(vars.w_idx);
    this->wgt_len(vars.wgt_len);
    this->inp_len(vars.inp_len);
    this->wgt_fifo(vars.wgt_fifo);
    this->inp_fifo(vars.inp_fifo);
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

    SC_CTHREAD(LoadInputs, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Post, clock);
    reset_signal_is(reset, true);

#pragma HLS array_partition variable = out complete dim = 0
#pragma HLS array_partition variable = g complete dim = 0
#pragma HLS array_partition variable = r complete dim = 0
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

  void start_compute(int index, unsigned int w_idx, unsigned int depth,
                     int ra) {
    if (index == 0) {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
      vars_0.compute.write(true);
    } else if (index == 1) {
      vars_1.ra.write(ra);
      vars_1.depth.write(depth);
      vars_1.w_idx.write(w_idx);
      vars_1.compute.write(true);
    } else if (index == 2) {
      vars_2.ra.write(ra);
      vars_2.depth.write(depth);
      vars_2.w_idx.write(w_idx);
      vars_2.compute.write(true);
    } else if (index == 3) {
      vars_3.ra.write(ra);
      vars_3.depth.write(depth);
      vars_3.w_idx.write(w_idx);
      vars_3.compute.write(true);
    } else {
      vars_0.ra.write(ra);
      vars_0.depth.write(depth);
      vars_0.w_idx.write(w_idx);
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

#endif // VMM_UNIT_H
