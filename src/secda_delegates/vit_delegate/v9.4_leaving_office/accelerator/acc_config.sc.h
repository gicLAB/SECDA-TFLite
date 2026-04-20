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

//! Added from RPP
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define AXI_DWIDTH 32
#define AXI_DWIDTH_4 (32 / 4)
#define AXI_TYPE sc_uint
#define s_mdma multi_dma<AXI_DWIDTH, 0>
#define mm_buf mm_buffer<unsigned long long>
#define a_ctrl acc_ctrl<int>
// !2
#define DMA_IN_BUF_SIZE_4 (dma_out1 - dma_in0)
#define NO_OF_DATA_CHANNELS 4
#define DMA_WGT_SIZE_4 (DMA_IN_BUF_SIZE_4 - DMA_SCRATCH_SIZE_4)
#define DMA_SCRATCH_SIZE_4 0x00040000 // 256KB , for 1 DMA channel


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

#define pkT 8
#define pkF 16

#define pnF pn_block     // Full pn
#define pnT pnF / 2 // half pn
#define pnQ pnT / 2 // quarter pn

#define pmF pm_block     // Full pm
#define pmT pmF / 2 // half pm
#define pmQ pmT / 2 // quarter pm

#define pn_block 64
#define pm_block 64

#define NUM_CORES 3

#define MODE_DENSE 0 // Broadcast inputs, split weights (fat layers)
#define MODE_MOBILE 1 // Broadcast weights, split inputs (skinny/square latyers)

#define max_pk 1024


//==============================================================================
// SystemC Specfic SIM/HW Configurations
//==============================================================================
#if defined(SYSC) || defined(__SYNTHESIS__)
#include <systemc.h>

#ifndef __SYNTHESIS__
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_profiler/profiler.h"
#define DWAIT(x) wait(x)

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else // !VERBOSE_ACC
#define ALOG(x)
#endif

typedef _BDATA<AXI_DWIDTH, AXI_TYPE> ADATA;


#else // __SYNTHESIS__
#include "sysc_types.h"
#define ALOG(x)

#define DWAIT(x)
struct _NDATA {
  AXI_TYPE<AXI_DWIDTH> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _NDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
  void pack(ACC_DTYPE<AXI_DWIDTH_4> a1, ACC_DTYPE<AXI_DWIDTH_4> a2,
            ACC_DTYPE<AXI_DWIDTH_4> a3, ACC_DTYPE<AXI_DWIDTH_4> a4) {
    data.range(7, 0) = a1;
    data.range(15, 8) = a2;
    data.range(23, 16) = a3;
    data.range(31, 24) = a4;
  }
};

typedef _NDATA ADATA;

#endif

#endif
#endif
