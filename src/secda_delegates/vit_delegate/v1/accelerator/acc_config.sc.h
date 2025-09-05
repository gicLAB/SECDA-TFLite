#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME VIT_ACC

// Address mapping for the accelerator and DMA
// #define acc_address 0x43C00000
// #define dma_addr0 0x40400000
// #define dma_in0 0x16000000
// #define dma_out0 0x16800000
// #define DMA_BL 4194304

#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
#define dma_in0 0x16000000
#define dma_in1 0x18000000
#define dma_in2 0x1a000000
#define dma_in3 0x1c000000
#define dma_out0 0x16800000
#define dma_out1 0x18800000
#define dma_out2 0x1a800000
#define dma_out3 0x1c800000
#define DMA_BL 4194304

#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define STOPPER -1

#define IN_BUF_LEN 4096
#define WE_BUF_LEN 8192
#define SUMS_BUF_LEN 1024

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648

#define MAX8 127
#define MIN8 -128
#define MAX32 2147483647
#define MIN32 -2147483648

#define pkT 32
#define pkF 64

#define pnF 128     // Full pn
#define pnT pnF / 2 // half pn
#define pnQ pnT / 2 // quarter pn

#define pmF 128     // Full pm
#define pmT pmF / 2 // half pm
#define pmQ pmT / 2 // quarter pm

//==============================================================================
// SystemC Specfic SIM/HW Configurations
//==============================================================================
#if defined(SYSC) || defined(__SYNTHESIS__)
#include <systemc.h>

#ifndef __SYNTHESIS__
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_integrator/sysc_types.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#define DWAIT(x) wait(x)

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else // !VERBOSE_ACC
#define ALOG(x)
#endif

#else // __SYNTHESIS__

#define DWAIT(x)
typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;
#endif

#endif
#endif
