#ifndef ACCNAME_H
#define ACCNAME_H

#include <systemc.h>

#include "acc_config.sc.h"

// #define __SYNTHESIS__

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  // ================================================= //
  // Global ports
  // ================================================= //

  // Control ports
  CTRL_Define_Ports;

  // Data ports
  AXI4M_Bus_Port(unsigned long long, insn);
  AXI4M_Bus_Port(unsigned long long, input);
  AXI4M_Bus_Port(unsigned long long, weight);
  AXI4M_Bus_Port(unsigned long long, bias);
  AXI4M_Bus_Port(unsigned int, output);

  sc_in<unsigned int> insn_count;
  sc_in<int> crf;
  sc_in<int> crx;
  sc_in<int> ra;
  sc_in<int> depth;

  // ================================================= //
  // Global variables
  // ================================================= //

  sc_uint<64> wgt_insn1;
  sc_uint<64> wgt_insn2;
  sc_uint<64> inp_insn1;
  sc_uint<64> inp_insn2;
  sc_uint<64> bias_insn1;
  sc_uint<64> bias_insn2;
  int ra_val;
  int crf_val;
  int crx_val;
  sc_int<64> pl;
  sc_int<32> pr;
  sc_int<32> msk;
  sc_int<32> sm;

  // ================================================= //
  // Global buffers
  // ================================================= //

#ifndef __SYNTHESIS__
  wgt_bt *wgt_mem1 = new wgt_bt[WGT_DEPTH];
  wgt_bt *wgt_mem2 = new wgt_bt[WGT_DEPTH];
  wgt_bt *wgt_mem3 = new wgt_bt[WGT_DEPTH];
  wgt_bt *wgt_mem4 = new wgt_bt[WGT_DEPTH];
  inp_bt *inp_mem1 = new inp_bt[INP_DEPTH];
  inp_bt *inp_mem2 = new inp_bt[INP_DEPTH];
  inp_bt *inp_mem3 = new inp_bt[INP_DEPTH];
  inp_bt *inp_mem4 = new inp_bt[INP_DEPTH];
  acc_bt *acc_mem = new acc_bt[ACC_DEPTH];
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

  // ================================================= //
  // Global signals
  // ================================================= //
  DEFINE_SC_SIGNAL(bool, wgt_load);
  DEFINE_SC_SIGNAL(bool, inp_load);
  DEFINE_SC_SIGNAL(bool, bias_load);
  DEFINE_SC_SIGNAL(bool, gemm_wait);
  DEFINE_SC_SIGNAL(bool, schedule);
  DEFINE_SC_SIGNAL(bool, storing);
  DEFINE_SC_SIGNAL(bool, loading);

  // Scheduler
  DEFINE_SC_SIGNAL(unsigned int, wgt_block);
  DEFINE_SC_SIGNAL(unsigned int, inp_block);
  DEFINE_SC_SIGNAL(unsigned int, depth_val);

  // GEMM Unit signals
  DEFINE_SC_SIGNAL(unsigned int, wp_val);
  DEFINE_SC_SIGNAL(unsigned int, ip_val);

  // Store Unit signals
  DEFINE_SC_SIGNAL(unsigned int, store_doffset);
  DEFINE_SC_SIGNAL(unsigned int, store_dstride);
  DEFINE_SC_SIGNAL(unsigned int, m_off);
  DEFINE_SC_SIGNAL(unsigned int, n_off);
  DEFINE_SC_SIGNAL(bool, fetch_resetted);

  // ================================================= //
  // Functions
  // ================================================= //

  sc_int<32> mul_s8(sc_int<8>, sc_int<8>);

  int Quantised_Multiplier(int, int, int);

  // ================================================= //
  // HW Threads
  // ================================================= //

  void Fetch();

  void Load_weights();

  void Load_inputs();

  void Load_bias();

  void Scheduler();

  void Compute();

  void Store();

#ifndef __SYNTHESIS_
  void Simulation_Profiler();
#endif

  // ================================================= //
  // Submodules
  // ================================================= //

  // ================================================= //
  // Profiling variable
  // ================================================= //

#ifndef __SYNTHESIS__
  // Profiling variable
  ClockCycles *total_cycles = new ClockCycles("total_cycles", true);
  DataCount *macs = new DataCount("macs");
  DataCount *ins_count = new DataCount("ins_count");
  std::vector<Metric *> profiling_vars = {total_cycles, macs, ins_count};
#endif

  // ================================================= //
  // HWC
  // ================================================= //

  // ================================================= //

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_)
      : sc_module(name_), insn_port("insn_port"), input_port("input_port"),
        weight_port("weight_port"), bias_port("bias_port"),
        output_port("output_port") {
    SC_CTHREAD(Fetch, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Load_weights, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Load_inputs, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Load_bias, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Store, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Scheduler, clock.pos());
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(Simulation_Profiler, clock.pos());
    reset_signal_is(reset, true);
#endif

    CTRL_PragGroup;
    AXI4M_PragAddr(insn);
    AXI4M_PragAddr(input);
    AXI4M_PragAddr(weight);
    AXI4M_PragAddr(bias);
    AXI4M_PragAddr(output);
    CTRL_Prag(insn_count);
    CTRL_Prag(crf);
    CTRL_Prag(crx);
    CTRL_Prag(ra);
    CTRL_Prag(depth);
  }
};

#endif // ACCNAME_H
