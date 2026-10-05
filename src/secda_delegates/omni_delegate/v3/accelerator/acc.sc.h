#ifndef ACCNAME_H
#define ACCNAME_H

// COPILOT NOTE: Added comment. In this variant, SystemC types are expected to
// come through acc_config.sc.h (which selects sim/synth headers by mode), so
// this header keeps includes minimal.
#include "acc_config.sc.h"
#include "omni_pe.sc.h"

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  // ================================================= //
  // Global ports
  // ================================================= //

  // Data ports
  sc_fifo_in<ADATA> din1;
  sc_fifo_out<ADATA> dout1;
  sc_out<int> computeSS;

  // ================================================= //
  // Global variables
  // ================================================= //
  int lshift;
  int in1_off;
  int in1_sv;
  int in1_mul;
  int in2_off;
  int in2_sv;
  int in2_mul;
  int out1_off;
  int out1_sv;
  int out1_mul;
  int qa_max;
  int qa_min;

  // V3: Added Fusion Routing Flags
  int src_route;
  int dst_route;

  // ================================================= //
  // Global buffers
  // ================================================= //

  // V3: Added on-chip scratchpad for intermediate feature maps
  // Configured here for 512KB (131,072 x 32-bit integers)
  // Adjust FUSION_BUFFER_DEPTH based on maximum expected tensor size
  static const int FUSION_BUFFER_DEPTH = 131072;
  int fusion_scratchpad[FUSION_BUFFER_DEPTH];


  // ================================================= //
  // Global signals
  // ================================================= //
  DEFINE_SC_SIGNAL(int, computeS);

  // ================================================= //
  // Functions
  // ================================================= //

  ACC_DTYPE<32> Clamp_Combine(int, int, int, int, int, int);

  void send_parameters_omni_PE(int, sc_fifo_in<ADATA> *);

  // ================================================= //
  // HW Threads
  // ================================================= //

  void Compute();

#ifndef __SYNTHESIS__
  void Simulation_Profiler();
#endif

  // ================================================= //
  // Submodules
  // ================================================= //

  struct omni_pe_var_array omni_pe_array;

  // ================================================= //
  // Profiling variable
  // ================================================= //

#ifndef __SYNTHESIS__
  ClockCycles *total_cycles = new ClockCycles("total_cycles", true);
  ClockCycles *active_cycles = new ClockCycles("active_cycles", true);
  SignalTrack *comS = new SignalTrack("T_compute", true);
  std::vector<Metric *> profiling_vars = {total_cycles, active_cycles, comS};
#endif
  // ================================================= //

  // ================================================= //
  // HWC
  // ================================================= //

  // ================================================= //

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {

    // Connect omni_PE ports
    omni_pe_array.init(clock, reset);

    // V3: Added Scratchpad Memroy
    // Map fusion scratchpad to BRAM
    // NOTE: If BRAM capacity exceeded on target board change core=RAM_2P_BRAM to core=RAM_2P_URAM
    #pragma HLS resource variable=fusion_scratchpad core=RAM_2P_BRAM

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

#ifndef __SYNTHESIS__
    SC_CTHREAD(Simulation_Profiler, clock);
    reset_signal_is(reset, true);
#endif

    SLV_Prag(computeSS);
    AXI4S_In_Prag(din1);
    AXI4S_Out_Prag(dout1);
  }
};

#endif