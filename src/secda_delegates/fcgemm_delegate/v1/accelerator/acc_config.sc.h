#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME FC_ACC_v2_0
#define SUBMODULENAME gemm_pe

//==============================================================================
// Hardware Constants
//==============================================================================
// Define any Hardware specific constants for the accelerator
// These constants will be accessible in the driver
// These constants will be used to generate the hardware

#define PAGE_SIZE getpagesize()

// Address mapping for the accelerator and DMA
#define MM_BL 4194304
#define dma_addr0 0x40400000
#define acc_address 0x43C00000
#define insn_address 0x16000000
#define in_address 0x17000000
#define wgt_address 0x18000000
#define out_address 0x19000000
#define bias_address 0x1a000000

// Data types
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define AXI_DWIDTH 32
#define AXI_TYPE sc_uint

// ACC Specific Constants

// Buffer sizes
#define INP_ACCESS 8
#define WGT_ACCESS 8
#define ACC_ACCESS 2
#define INP_MEMS 4
#define WGT_MEMS 4
#define ACC_MEMS 1
#define INP_SIZE (INP_DEPTH * INP_ACCESS * INP_MEMS)
#define WGT_SIZE (WGT_DEPTH * WGT_ACCESS * WGT_MEMS)
#define ACC_SIZE (ACC_DEPTH * ACC_ACCESS * ACC_MEMS)
#define INP_DEPTH 4096
#define WGT_DEPTH 8192
#define ACC_DEPTH 8192

#define SC_INP_ELEM_BYTES_RATIO 4
#define MAX8 127
#define MIN8 -128
#define MAX32 2147483647
#define MIN32 -2147483648

#define DIVMAX 2147483648
#define POS 1073741824
#define NEG -1073741823

#define s_mdma multi_dma<AXI_DWIDTH, 0>

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
#define DPROF(x) x

#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else // !VERBOSE_ACC
#define ALOG(x)
#endif

typedef _BDATA<AXI_DWIDTH, AXI_TYPE> ADATA;

#else // __SYNTHESIS__
#include "AXI4_if.h"
#include "sysc_types.h"
#define ALOG(x)

struct _NDATA {
  AXI_TYPE<AXI_DWIDTH> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _NDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
};

typedef _NDATA ADATA;

#define DWAIT(x)
#define DPROF(x)
#endif

typedef unsigned long long inp_bt;
typedef unsigned long long wgt_bt;
typedef int out_bt;
typedef unsigned long long acc_bt;
typedef sc_int<8> dat_t;
typedef sc_int<32> acc_t;

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

//==============================================================================
// HW Submodule Construction SIM/HW Structs
//==============================================================================

// UNUSED

//==============================================================================

#endif // defined(SYSC) || defined(__SYNTHESIS__)
#endif // ACC_CONFIG_H