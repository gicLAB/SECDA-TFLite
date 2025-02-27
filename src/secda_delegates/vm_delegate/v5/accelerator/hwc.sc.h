#ifndef HWC_H
#define HWC_H

#include <systemc.h>

#define PRAGMA(X) _Pragma(#X)

#define HWC_AXI4LiteS(name, signame)                                           \
  PRAGMA(HLS resource core = AXI4LiteS metadata = "-bus_bundle hwc" variable = \
             name##_##signame)

#define HWC_Define_Signals(name)                                               \
  sc_signal<unsigned int> sig_##name##_sts;                                    \
  sc_signal<unsigned int> sig_##name##_co;                                     \
  sc_signal<unsigned int> sig_##name##_so;

#define HWC_Bind_Signals(name)                                                 \
  acc->name##_sts(scs->sig_##name##_sts);                                      \
  acc->name##_co(scs->sig_##name##_co);                                        \
  acc->name##_so(scs->sig_##name##_so);



// This defines the ports for the HWC
#define HWC_Define(name)                                                       \
  sc_in<unsigned int> name##_sts;                                              \
  sc_signal<unsigned int> name##_si;                                           \
  sc_out<unsigned int> name##_co;                                              \
  sc_out<unsigned int> name##_so;                                              \
  unsigned int name##_cycles;

// This defines the logic for the HWC
#define HWC_Logic(name)                                                        \
  unsigned int name##_state = 0;                                                \
  name##_state = name##_si.read();                                              \
  if (hwc_reset.read()) {                                                      \
    name##_cycles = 0;                                                         \
    name##_co.write(name##_cycles);                                            \
    name##_so.write(name##_state);                                              \
  } else if (name##_state == name##_sts) {                                      \
    name##_cycles++;                                                           \
    name##_co.write(name##_cycles);                                            \
    name##_so.write(name##_state);                                              \
  } else {                                                                     \
    name##_co.write(name##_cycles);                                            \
    name##_so.write(name##_state);                                              \
  }

#ifdef __SYNTHESIS__
// This defines CTHREADs which are monitored an HWC
#define HWC_CTHREAD(name) HWC_Define(name) \
  void name();

// This defines the AXI4LiteS ports for the HWC
#define HWC_PragGroup(name)                                                    \
  HWC_AXI4LiteS(name, sts) HWC_AXI4LiteS(name, co) HWC_AXI4LiteS(name, so)
#else
#define HWC_CTHREAD(name) void name##();

#define HWC_PragGroup(name)
#endif
#endif // HWC_H

// #define HWC_MAIN(name) \
// 	void HW_MAIN() { \
// 		wait(); \
// 		while (true) { \
// 			{ \
// 	PRAGMA(HLS LATENCY max = 0 min = 0 ) \
// 	PRAGMA(HLS protocol fixed) \
// 				HWC_Logic(A); \
// 				HWC_Logic(B); \
// 				HWC_Logic(C); \
// 				DWAIT(); \
// 			} \
// 		} \
// 	}
