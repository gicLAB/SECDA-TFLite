#ifndef VTA_H
#define VTA_H

#include <systemc.h>
#include "AXI4_if.h"

#include <iostream>
using namespace std;

#ifndef __SYNTHESIS__
// #define DWAIT(x) wait(x)
#define DWAIT() wait(1)
#define DWAIT(x) wait(1)
#define DPROF(x) x
#else
#define DWAIT(x)
#define DPROF(x)
#endif

#ifndef __SYNTHESIS__
//#define VLOG(X) cout X
#define VLOG(X)
#else
#define VLOG(X)
#endif

// typedef sc_int<64> inp_bt;
// typedef sc_int<64> wgt_bt;
// typedef sc_int<64> acc_bt;
typedef unsigned long long inp_bt;
typedef unsigned long long wgt_bt;
//typedef sc_int<32> out_bt;
typedef int out_bt;

typedef unsigned long long acc_bt;
//typedef sc_int<64> acc_bt;

typedef sc_int<8> dat_t;
typedef sc_int<32> acc_t;

#define INP_ACCESS 8
#define WGT_ACCESS 8
#define ACC_ACCESS 2

#define INP_DEPTH 4096
#define WGT_DEPTH 8192
#define ACC_DEPTH 8192

//#define INP_DEPTH 8192
//#define WGT_DEPTH 8192
//#define ACC_DEPTH 2048

#define INP_SIZE (INP_DEPTH * INP_ACCESS * INP_MEMS)
#define WGT_SIZE (WGT_DEPTH * WGT_ACCESS * WGT_MEMS)
#define ACC_SIZE (ACC_DEPTH * ACC_ACCESS * ACC_MEMS)

#define SC_INP_ELEM_BYTES_RATIO 4
#define MAX8 127
#define MIN8 -128
#define MAX32 2147483647
#define MIN32 -2147483648

#define DIVMAX 2147483648
#define POS 1073741824
#define NEG -1073741823

struct opcode {
  unsigned long long p1;
  unsigned long long p2;

  int dstride;
  int x_size;
  int y_size;
  int doffset;
  int op;

  opcode(sc_uint<64> _p1, sc_uint<64> _p2) {
    p1 = _p1;
    p2 = _p2;

    dstride = _p1.range(63, 32);
    x_size = _p1.range(31, 16);
    y_size = _p1.range(15, 0);

    doffset = _p2.range(63, 32);
    op = _p2.range(31, 0);
  }
};

#define ACCNAME FC_ACC
SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<unsigned int> start_acc;
  sc_out<unsigned int> done_acc;
  sc_in<bool> reset_acc;

  sc_in<unsigned int> insn_count;
  sc_in<unsigned int> insn_addr;
  sc_in<unsigned int> input_addr;
  sc_in<unsigned int> weight_addr;
  sc_in<unsigned int> bias_addr;
  sc_in<unsigned int> output_addr;

  sc_in<int> crf;
  sc_in<int> crx;
  sc_in<int> ra;
  sc_in<int> depth;

  AXI4M_bus_port<unsigned long long> insn_port;
  AXI4M_bus_port<unsigned long long> input_port;
  AXI4M_bus_port<unsigned long long> weight_port;
  AXI4M_bus_port<unsigned long long> bias_port;
  AXI4M_bus_port<unsigned int> out_port;

  // Instantiate memories
#ifndef __SYNTHESIS__
  wgt_bt* wgt_mem1 = new wgt_bt[WGT_DEPTH];
  wgt_bt* wgt_mem2 = new wgt_bt[WGT_DEPTH];
  wgt_bt* wgt_mem3 = new wgt_bt[WGT_DEPTH];
  wgt_bt* wgt_mem4 = new wgt_bt[WGT_DEPTH];
  inp_bt* inp_mem1 = new inp_bt[INP_DEPTH];
  inp_bt* inp_mem2 = new inp_bt[INP_DEPTH];
  inp_bt* inp_mem3 = new inp_bt[INP_DEPTH];
  inp_bt* inp_mem4 = new inp_bt[INP_DEPTH];
  acc_bt* acc_mem = new acc_bt[ACC_DEPTH];
#else
  wgt_bt wgt_mem1[WGT_DEPTH];
  wgt_bt wgt_mem2[WGT_DEPTH];
  wgt_bt wgt_mem3[WGT_DEPTH];
  wgt_bt wgt_mem4[WGT_DEPTH];
  inp_bt inp_mem1[INP_DEPTH];
  inp_bt inp_mem2[INP_DEPTH];
  inp_bt inp_mem3[INP_DEPTH];
  inp_bt inp_mem4[INP_DEPTH];
  acc_bt acc_mem[ACC_DEPTH];
#endif

  out_bt out_mem[4][4];

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> wgt_load;
  sc_signal<bool, SC_MANY_WRITERS> inp_load;
  sc_signal<bool, SC_MANY_WRITERS> bias_load;
  sc_signal<bool, SC_MANY_WRITERS> gemm_wait;
  sc_signal<bool, SC_MANY_WRITERS> schedule;
  sc_signal<bool, SC_MANY_WRITERS> storing;
#else
  sc_signal<bool> wgt_load;
  sc_signal<bool> inp_load;
  sc_signal<bool> bias_load;
  sc_signal<bool> gemm_wait;
  sc_signal<bool> schedule;
  sc_signal<bool> storing;
#endif
  sc_signal<bool> loading;


  sc_uint<64> wgt_insn1;
  sc_uint<64> wgt_insn2;

  sc_uint<64> inp_insn1;
  sc_uint<64> inp_insn2;

  sc_uint<64> bias_insn1;
  sc_uint<64> bias_insn2;

//  sc_signal<unsigned int> bias_load_doffset;
//  sc_signal<unsigned int> bias_load_dstride;
//  sc_signal<unsigned int> bias_minc;
//  sc_signal<unsigned int> bias_load_length;


  // Scheduler
  sc_signal<unsigned int> wgt_block;
  sc_signal<unsigned int> inp_block;
  sc_signal<unsigned int> depth_val;

  // GEMM Unit signals
  sc_signal<unsigned int> wp_val;
  sc_signal<unsigned int> ip_val;

  // Store Unit signals
  sc_signal<unsigned int> store_doffset;
  sc_signal<unsigned int> store_dstride;
  sc_signal<unsigned int> m_off;
  sc_signal<unsigned int> n_off;


  sc_signal<bool> fetch_resetted;

  int start_count;
  int done_count;
  int ra_val;
  int crf_val;
  int crx_val;
  bool resetted;

	sc_int<64> pl;
	sc_int<32> pr;
	sc_int<32> msk;
	sc_int<32> sm;




//  // Profiling variable
//  ClockCycles* per_batch_cycles = new ClockCycles("per_batch_cycles", true);
//  std::vector<Metric*> profiling_vars = {per_batch_cycles};
//  int layer = 0;
//  int pc = 0;


	sc_int<32> mul_s8(sc_int<8>,sc_int<8>);

  int Quantised_Multiplier(int, int, int);

  void fetch();

  void load_weights();
  void load_inputs();
  void load_bias();

  void scheduler();

  void compute();

  void store();

  void tracker();

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_), // @suppress("Class members should be properly initialized")
  insn_port("insn_port"),
  input_port("input_port"),
  weight_port("weight_port"),
  bias_port("bias_port"),
  out_port("out_port")

  
  {
    SC_CTHREAD(fetch, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(load_weights, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(load_inputs, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(load_bias, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(store, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(scheduler, clock.pos());
    reset_signal_is(reset, true);

    // SC_CTHREAD(tracker, clock.pos());
    // reset_signal_is(reset, true);

#pragma HLS RESET variable = reset
  }
};

#endif  // VTA_H
