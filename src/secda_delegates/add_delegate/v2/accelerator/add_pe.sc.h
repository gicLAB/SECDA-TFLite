#ifndef ADD_PE_H
#define ADD_PE_H

#include "acc_config.sc.h"

SC_MODULE(SUBMODULENAME) {
  // Declare I/O ports and fifos for the hardware submodule
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<bool> start;
  sc_out<bool> done;
  sc_in<int> length;
  sc_in<int> lshift;
  sc_in<int> in1_off;
  sc_in<int> in1_sv;
  sc_in<int> in1_mul;
  sc_in<int> in2_off;
  sc_in<int> in2_sv;
  sc_in<int> in2_mul;
  sc_in<int> out1_off;
  sc_in<int> out1_sv;
  sc_in<int> out1_mul;

  sc_fifo_in<int> input_fifo;
  sc_fifo_out<int> output_fifo;

  // Declare buffers for the hardware submodule
  sc_int<8> i1mem[4];
  sc_int<8> i2mem[4];
  int s_in1[4];
  int s_in2[4];
  int sum[4];
  int f_out[4];

  // Functions
  int Quantised_Multiplier(int x, int qm, int shift) {
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
      pl = 1;
      pr = -shift;
      msk = (1 << -shift) - 1;
      sm = msk >> 1;
    }
    sc_int<64> val = x * pl;
    if (val > MAX) val = MAX;
    if (val < MIN) val = MIN;
    sc_int<64> val_2 = val * qm;
    sc_int<32> add_1;
    add_1 = (val_2 + POS) / DIVMAX;
    if (val_2 < 0) add_1 = (val_2 + NEG) / DIVMAX;
    sc_int<32> val_3 = add_1;
    val_3 = val_3 >> pr;
    sc_int<32> add_2 = add_1 & msk;
    sc_int<32> add_3 = (add_1 < 0) & 1;
    sc_int<32> add_4 = sm + add_3;
    sc_int<32> add_5 = ((add_2 > add_4) & 1);
    sc_int<32> result_32 = val_3 + add_5;
    return result_32;
  }

  // HW Threads
  void Compute() {
    ACC_DTYPE<32> i1;
    ACC_DTYPE<32> i2;
    done.write(0);
    while (true) {
      i1 = input_fifo.read();
      i2 = input_fifo.read();
      i1mem[0] = i1.range(7, 0);
      i1mem[1] = i1.range(15, 8);
      i1mem[2] = i1.range(23, 16);
      i1mem[3] = i1.range(31, 24);
      i2mem[0] = i2.range(7, 0);
      i2mem[1] = i2.range(15, 8);
      i2mem[2] = i2.range(23, 16);
      i2mem[3] = i2.range(31, 24);
      s_in1[0] = (i1mem[0] + in1_off) * lshift;
      s_in1[1] = (i1mem[1] + in1_off) * lshift;
      s_in1[2] = (i1mem[2] + in1_off) * lshift;
      s_in1[3] = (i1mem[3] + in1_off) * lshift;
      s_in2[0] = (i2mem[0] + in2_off) * lshift;
      s_in2[1] = (i2mem[1] + in2_off) * lshift;
      s_in2[2] = (i2mem[2] + in2_off) * lshift;
      s_in2[3] = (i2mem[3] + in2_off) * lshift;
      s_in1[0] = Quantised_Multiplier(s_in1[0], in1_mul, in1_sv);
      s_in1[1] = Quantised_Multiplier(s_in1[1], in1_mul, in1_sv);
      s_in1[2] = Quantised_Multiplier(s_in1[2], in1_mul, in1_sv);
      s_in1[3] = Quantised_Multiplier(s_in1[3], in1_mul, in1_sv);
      s_in2[0] = Quantised_Multiplier(s_in2[0], in2_mul, in2_sv);
      s_in2[1] = Quantised_Multiplier(s_in2[1], in2_mul, in2_sv);
      s_in2[2] = Quantised_Multiplier(s_in2[2], in2_mul, in2_sv);
      s_in2[3] = Quantised_Multiplier(s_in2[3], in2_mul, in2_sv);
      sum[0] = s_in1[0] + s_in2[0];
      sum[1] = s_in1[1] + s_in2[1];
      sum[2] = s_in1[2] + s_in2[2];
      sum[3] = s_in1[3] + s_in2[3];
      f_out[0] = Quantised_Multiplier(sum[0], out1_mul, out1_sv) + out1_off;
      f_out[1] = Quantised_Multiplier(sum[1], out1_mul, out1_sv) + out1_off;
      f_out[2] = Quantised_Multiplier(sum[2], out1_mul, out1_sv) + out1_off;
      f_out[3] = Quantised_Multiplier(sum[3], out1_mul, out1_sv) + out1_off;
      output_fifo.write(f_out[0]);
      output_fifo.write(f_out[1]);
      output_fifo.write(f_out[2]);
      output_fifo.write(f_out[3]);
      wait();
    }
  };

  // This binds the submodule ports to the accelerator signals to the submodule
  void init(sc_in<bool> & clock, sc_in<bool> & reset, add_pe_vars & vars) {
    this->clock(clock);
    this->reset(reset);

    this->start(vars.start);
    this->done(vars.done);
    this->length(vars.length);
    this->lshift(vars.lshift);
    this->in1_off(vars.in1_off);
    this->in1_sv(vars.in1_sv);
    this->in1_mul(vars.in1_mul);
    this->in2_off(vars.in2_off);
    this->in2_sv(vars.in2_sv);
    this->in2_mul(vars.in2_mul);
    this->out1_off(vars.out1_off);
    this->out1_sv(vars.out1_sv);
    this->out1_mul(vars.out1_mul);

    this->input_fifo(vars.input_fifo);
    this->output_fifo(vars.output_fifo);
  }

  SC_HAS_PROCESS(SUBMODULENAME);

  SUBMODULENAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    // Define any HLS pragma for the submodule declared here
  }
};

// This initializes the submodule array
// It provides HLS support methods to interact with the submodules
struct add_pe_var_array {
  add_pe_vars vars_0;
  add_pe_vars vars_1;
  SUBMODULENAME HW0;
  SUBMODULENAME HW1;

#ifndef __SYNTHESIS__
  add_pe_var_array() : vars_0(16, 0), vars_1(16, 1), HW0("HW0"), HW1("HW1") {}
#else
  add_pe_var_array() : vars_0(16), vars_1(16), HW0("HW0"), HW1("HW1") {}
#endif

  add_pe_vars& operator[](int i) {
    if (i == 0)
      return vars_0;
    else if (i == 1)
      return vars_1;
    else
      return vars_0;
  }

  void init(sc_in<bool>& clock, sc_in<bool>& reset) {
    HW0.init(clock, reset, vars_0);
    HW1.init(clock, reset, vars_1);
  }

  void input_fifo_write(int i, int val) {
    if (i == 0)
      vars_0.input_fifo.write(val);
    else if (i == 1)
      vars_1.input_fifo.write(val);
    else
      vars_0.input_fifo.write(val);
  }

  int output_fifo_read(int i) {
    if (i == 0)
      return vars_0.output_fifo.read();
    else if (i == 1)
      return vars_1.output_fifo.read();
    else
      return vars_0.output_fifo.read();
  }
};
#endif  // ADD_PE_H