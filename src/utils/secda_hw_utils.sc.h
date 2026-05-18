#ifndef SECDA_HW_UTILS_SC_H
#define SECDA_HW_UTILS_SC_H

#include <systemc.h>

#ifdef SYSC
#include "../ap_sysc/AXI4_if.h"
#include "../ap_sysc/ap_mem_if.h"
#else
#include "AXI4_if.h"
#include "ap_mem_if.h"
#endif

// ================================================================
// Misc || Synthesizable and Simulation
// ================================================================

#ifndef __SYNTHESIS__

// Define macros for simulation
#define PRAGMA(X)
#define DEFINE_SC_SIGNAL(type, name) sc_signal<type, SC_MANY_WRITERS> name;
#define DEFINE_SHAKE_SIGNALS(name)                                             \
  sc_signal<bool, SC_MANY_WRITERS> start_##name;                               \
  sc_signal<bool, SC_MANY_WRITERS> done_##name;

#define DEFINE_STATUS_SIGNALS(type, name)                                    \
  sc_signal<type, SC_MANY_WRITERS> name##S;                                    \
  sc_out<type> name##SS;

#define INITSIGPORT(X, SID) X((std::string(#X)).c_str())
#else

// Define macros for synthesis
#define DEFINE_SC_SIGNAL(type, name) sc_signal<type> name;
#define DEFINE_SHAKE_SIGNALS(name)                                             \
  sc_signal<bool> start_##name;                                                \
  sc_signal<bool> done_##name;

#define DEFINE_STATUS_SIGNALS(type, name)                                    \
  sc_signal<type> name##S;                                                     \
  sc_out<type> name##SS;

#define INITSIGPORT(X, SID) X((std::string(#X) + std::to_string(SID)).c_str())
#define PRAGMA(X) _Pragma(#X)
#endif

#define Clock_Reset_Define                                                     \
  sc_clock clk_clock;                                                          \
  sc_signal<bool> sig_reset;

#define Clock_Reset_Bind(dut, scs)                                             \
  dut->clock(scs->clk_clock);                                                  \
  dut->reset(scs->sig_reset);

#define SIGWRITE(X, VAL)                                                       \
  X.write(VAL);                                                                \
  X##S.write(VAL);

#define HWC_SIG(X, VAL) X##_si.write(VAL);
// ================================================================
// Control Unit Macros || Synthesizable and Simulation
// ================================================================
static unsigned int CTRL_SIG_Counter = 0;

#define SLV_Prag(signame)                                                      \
  PRAGMA(HLS resource core = AXI4LiteS metadata =                              \
             "-bus_bundle slv0" variable = signame)

#define CTRL_Prag(signame)                                                     \
  PRAGMA(HLS resource core = AXI4LiteS metadata =                              \
             "-bus_bundle ctrl" variable = signame)

#define CTRL_Define_Ports                                                      \
  sc_in<bool> start;                                                           \
  sc_out<bool> done;

#define CTRL_Define_Signals                                                    \
  sc_signal<bool> sig_start;                                                   \
  sc_signal<bool> sig_done;

#define CTRL_Bind_Signals(dut, scs)                                            \
  dut->done(scs->sig_done);                                                    \
  dut->start(scs->sig_start);

#define CTRL_Bind_CtrlSignals(dut, ctrl)                                       \
  dut->done(ctrl->ctrl_sigs->sig_done);                                        \
  dut->start(ctrl->ctrl_sigs->sig_start);

#define CTRL_Bind_RegSignals(signame)                                          \
  acc->signame(ctrl->reg_sigs[CTRL_SIG_Counter++].sig);

#define CTRL_PragGroup CTRL_Prag(start) CTRL_Prag(done)

// ================================================================
// AXIMM Macros || Synthesizable and Simulation
// ================================================================

#define AXI4M_Bus_Port(type, name)                                             \
  AXI4M_bus_port<type> name##_port;                                            \
  sc_in<unsigned int> name##_addr;

#define AXI4M_PragAddr(name) CTRL_Prag(name##_addr)

// ================================================================
// AXIS Macros || Synthesizable and Simulation
// ================================================================

#define AXI4S_In_Prag(signame)                                                 \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle S_AXIS_DATA1" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_Out_Prag(signame)                                                \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle M_AXIS_DATA1" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_In_Prag1(signame)                                                \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle S_AXIS_DATA1" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_Out_Prag1(signame)                                               \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle M_AXIS_DATA1" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_In_Prag2(signame)                                                \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle S_AXIS_DATA2" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_Out_Prag2(signame)                                               \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle M_AXIS_DATA2" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_In_Prag3(signame)                                                \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle S_AXIS_DATA3" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_Out_Prag3(signame)                                               \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle M_AXIS_DATA3" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_In_Prag4(signame)                                                \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle S_AXIS_DATA4" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_Out_Prag4(signame)                                               \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle M_AXIS_DATA4" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

//==================
//==================
// Signal Macros
//==================
#define Sig_Start(start_sig, done_sig)                                         \
  start_sig.write(true);                                                       \
  wait();                                                                      \
  while (!done_sig.read()) wait();                                             \
  start_sig.write(false);                                                      \
  wait();

#define Sig_Wait(start_sig)                                                    \
  while (!start_sig.read()) wait();

#define Sig_Done(start_sig, done_sig)                                          \
  done_sig.write(true);                                                        \
  wait();                                                                      \
  while (start_sig.read()) wait();                                             \
  done_sig.write(false);                                                       \
  wait();

#define Shake_Reset_M(sig) start_##sig.write(0);
#define Shake_Reset_S(sig) done_##sig.write(0);

#define Shake_Start(sig)                                                       \
  start_##sig.write(true);                                                     \
  wait();                                                                      \
  while (!done_##sig.read()) wait();                                           \
  start_##sig.write(false);                                                    \
  wait();

#define Shake_Wait(sig)                                                        \
  while (!start_##sig.read()) wait();

#define Shake_Done(sig)                                                        \
  done_##sig.write(true);                                                      \
  wait();                                                                      \
  while (start_##sig.read()) wait();                                           \
  done_##sig.write(false);                                                     \
  wait();

#define SigOut_Write(X, VAL)                                                   \
  X.write(VAL);                                                                \
  X##S.write(VAL)

#define Sig_Write(X, Y) X.write(Y)

//==================

// ================================================================
// Hardware Utility Structs || Synthesizable and Simulation
// ================================================================

/*
BRAM module used to track memory accesses in the hardware.
*/
template <typename T, unsigned int W>
struct SBRAM {
  T data[W];
  int size;
  int access_count;
  int idx;
  SBRAM() {
    size = W;
    access_count = 0;
  }
  T &operator[](int i) {
    idx = i;
    return data[i];
  }
  int &operator=(int val) {
    data[idx] = val;
    return data[idx];
  }
  void write(int i, T val) { data[i] = val; }
  T read(int i) { return data[i]; }
};

/*
Hardware struct to contain output signal and port
*/
struct sc_out_sig {
  sc_out<int> oS;
  sc_signal<int> iS;
  void write(int x) {
    oS.write(x);
    iS.write(x);
  }
  int read() { return iS.read(); }
  void operator=(int x) { write(x); }
  void bind(sc_signal<int> &sig) { oS.bind(sig); }
  void operator()(sc_signal<int> &sig) { bind(sig); }
  void bind(sc_out<int> &sig) { oS.bind(sig); }
  void operator()(sc_out<int> &sig) { bind(sig); }
};

typedef struct _SDATA {
  sc_int<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _SDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} SDATA;

// ================================================================
// Simulation-only SystemC Modules
// ================================================================

#ifndef __SYNTHESIS__

SC_MODULE(ACC_CONTROL) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_out<bool> start;
  sc_in<bool> done;
  bool start_bool = false;

  void HandShake() {
    while (1) {
      while (!start_bool) wait();
      start.write(true);

      while (!done) wait();
      start.write(false);
      start_bool = false;
      sc_pause();
      while (done) wait();
    }
  };

  SC_HAS_PROCESS(ACC_CONTROL);

  ACC_CONTROL(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(HandShake, clock.pos());
    reset_signal_is(reset, true);
  }
};

SC_MODULE(HWC_RESETTER) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_out<bool> hwc_reset;
  bool hwc_reset_bool = false;

  void Reset() {
    wait();
    while (true) {
      if (hwc_reset_bool) {
        hwc_reset.write(true);
        wait();
        hwc_reset.write(false);
        hwc_reset_bool = false;
        sc_pause();
      }
      wait();
    }
  }

  SC_HAS_PROCESS(HWC_RESETTER);

  HWC_RESETTER(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Reset, clock.pos());
    reset_signal_is(reset, true);
  }
};

#endif // __SYNTHESIS__

  // ================================================================
  // Simulation API Structs
  // ================================================================

#ifndef __SYNTHESIS__

struct hwc_signals {
  sc_signal<unsigned int> sts;
  sc_signal<unsigned int> co;
  sc_signal<unsigned int> so;
  hwc_signals() {
    sts.write(0);
    co.write(0);
    so.write(0);
  }
};

struct reg_signals {
  sc_signal<unsigned int> sig;
  reg_signals() { sig.write(0); }
};

struct ctrl_signals {
  sc_signal<bool> sig_start;
  sc_signal<bool> sig_done;
  ctrl_signals() {
    sig_start.write(false);
    sig_done.write(false);
  }
};

struct clock_reset {
  sc_clock clk_clock;
  sc_signal<bool> sig_reset;
  clock_reset() : clk_clock("ClkClock") { sig_reset.write(false); }
};
#endif

// ================================================================
// Simulation Macros
// ================================================================

// ================================================================

#endif // SECDA_HW_UTILS_SC_H