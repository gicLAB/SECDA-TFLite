#ifndef HWC_H
#define HWC_H

#include <systemc.h>

#define PRAGMA(X) _Pragma(#X)

#define HWC_Define_Reset sc_signal<bool> sig_hwc_reset;
#define HWC_Bind_Reset acc->hwc_reset(scs->sig_hwc_reset);
#define HWC_Reset sc_in<bool> hwc_reset;
#define HWC_PragReset                                                          \
  PRAGMA(HLS resource core = AXI4LiteS metadata = "-bus_bundle hwc" variable = \
             hwc_reset)

#define HWC_AXI4LiteS(name, signame)                                           \
  PRAGMA(HLS resource core = AXI4LiteS metadata = "-bus_bundle hwc" variable = \
             name##_##signame)

// This defines the AXI4LiteS ports for the HWC
#define HWC_PragGroup(name)                                                    \
  HWC_AXI4LiteS(name, sts) HWC_AXI4LiteS(name, co) HWC_AXI4LiteS(name, so)

// This defines the sim HWC signals
#define HWC_Define_Signals(name)                                               \
  sc_signal<unsigned int> sig_##name##_sts;                                    \
  sc_signal<unsigned int> sig_##name##_co;                                     \
  sc_signal<unsigned int> sig_##name##_so;

// This binds the sim HWC signals to the ACC
#define HWC_Bind_Signals(name)                                                 \
  acc->name##_sts(scs->sig_##name##_sts);                                      \
  acc->name##_co(scs->sig_##name##_co);                                        \
  acc->name##_so(scs->sig_##name##_so);

// This defines the HWC ports in the ACC
#define HWC_Define(name)                                                       \
  sc_in<unsigned int> name##_sts;                                              \
  sc_signal<unsigned int> name##_si;                                           \
  sc_out<unsigned int> name##_co;                                              \
  sc_out<unsigned int> name##_so;                                              \
  unsigned int name##_cycles;

// This defines CTHREADs which are monitored by the HWC
#define HWC_CTHREAD(name) HWC_Define(name) void name();

// This defines the logic for the HWC
#define HWC_Logic(name)                                                        \
  unsigned int name##_state = 0;                                               \
  name##_state = name##_si.read();                                             \
  if (hwc_reset.read()) {                                                      \
    name##_cycles = 0;                                                         \
    name##_co.write(name##_cycles);                                            \
    name##_so.write(name##_state);                                             \
  } else if (name##_state == name##_sts) {                                     \
    name##_cycles++;                                                           \
    name##_co.write(name##_cycles);                                            \
    name##_so.write(name##_state);                                             \
  } else {                                                                     \
    name##_co.write(name##_cycles);                                            \
    name##_so.write(name##_state);                                             \
  }

#endif // HWC_H
