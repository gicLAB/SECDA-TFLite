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

template <typename T, unsigned int W>
struct sbram {
  T data[W];
  int size;
  int access_count;
  int idx;
  sbram() {
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

// sbam<sc_int<32>, 32> sram;

#define PRAGMA(X) _Pragma(#X)

#define SLV_Prag(signame)                                                     \
  PRAGMA(HLS resource core = AXI4LiteS metadata =                              \
             "-bus_bundle slv0" variable = signame)

             
#define CTRL_Prag(signame)                                                     \
  PRAGMA(HLS resource core = AXI4LiteS metadata =                              \
             "-bus_bundle ctrl" variable = signame)

#define AXI4S_In_Prag(signame)                                                 \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle S_AXIS_DATA1" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#define AXI4S_Out_Prag(signame)                                                \
  PRAGMA(HLS RESOURCE variable = signame core = AXI4Stream metadata =          \
             "-bus_bundle M_AXIS_DATA1" port_map = {                           \
                 {signame##_0 TDATA} {signame##_1 TLAST}})

#ifndef __SYNTHESIS__
#define DEFINE_SC_SIGNAL(type, name) sc_signal<type, SC_MANY_WRITERS> name;
#else
#define DEFINE_SC_SIGNAL(type, name) sc_signal<type> name;
#endif

#define CTRL_Define_Ports                                                      \
  sc_in<bool> start;                                                           \
  sc_out<bool> done;

#define CTRL_Define_Signals                                                    \
  sc_signal<bool> sig_start;                                                   \
  sc_signal<bool> sig_done;

#define CTRL_Bind_Signals(dut, scs)                                            \
  dut->done(scs->sig_done);                                                    \
  dut->start(scs->sig_start);

#define CTRL_PragGroup CTRL_Prag(start) CTRL_Prag(done)

#define AXI4M_Bus_Port(type, name)                                             \
  AXI4M_bus_port<type> name##_port;                                            \
  sc_in<unsigned int> name##_addr;

#define AXI4M_PragAddr(name) CTRL_Prag(name##_addr)
#endif // SECDA_HW_UTILS_SC_H