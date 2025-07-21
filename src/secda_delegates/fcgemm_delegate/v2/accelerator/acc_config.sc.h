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

//==============================================================================
// Address mapping for the accelerator and DMA
//==============================================================================

#ifdef KRIA
// KRIA
// Pre-Defined Address for Accelerator
#define acc_address 0x00A0000000
#define dma_addr0 0x00A0010000
#define dma_addr1 0x00A0020000
#define dma_addr2 0x00A0030000
#define dma_addr3 0x00A0040000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x0000000037400000
#define DMA_RANGE_END 0x00000000773FFFFF
#define DMA_RANGE_OFFSET 0xC00000         // 1.5MB
#define DMA_RANGE_SIZE 0x0000000040000000 // 1GB
#define DMA_IN_BUF_SIZE 0x20000000        // 32MB
#define DMA_OUT_BUF_SIZE 0x20000000       // 32MB

#define dma_in0 0x38000000
#define dma_in1 0x3A000000
#define dma_in2 0x3C000000
#define dma_in3 0x3E000000

#define dma_out0 0x39000000
#define dma_out1 0x3B000000
#define dma_out2 0x3D000000
#define dma_out3 0x40000000

#else
// Z1
// Pre-Defined Address for Accelerator
#define acc_ctrl_address 0x43C00000
#define acc_hwc_address 0x43C10000

#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000

#define dma_in0 0x18000000
#define dma_in1 0x1a000000
#define dma_in2 0x1c000000
#define dma_in3 0x1e000000
#define dma_out0 0x18800000
#define dma_out1 0x1a800000
#define dma_out2 0x1c800000
#define dma_out3 0x1e800000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x18000000
#define DMA_RANGE_END 0x1fffffff
#define DMA_RANGE_SIZE 0x8000000
#endif // KRIA

// AXIMM Constants
#ifdef KRIA
#define MM_BL 4194304 // 32MB
#define insn_address 0x38000000
#define in_address 0x39000000
#define wgt_address 0x3a000000
#define bias_address 0x3b000000
#define out_address 0x3c000000
#else
// Z1
#define MM_BL 4194304 // 32MB
#define insn_address 0x18000000
#define in_address 0x19000000
#define wgt_address 0x1a000000
#define bias_address 0x1b000000
#define out_address 0x1c000000
#endif

//==============================================================================
// Data types
//==============================================================================
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define AXI_DWIDTH 32
#define AXI_TYPE sc_uint
#define s_mdma multi_dma<AXI_DWIDTH, 0>
#define mm_buf mm_buffer<unsigned long long>
#define mm_buf2 mm_buffer<unsigned int>
#define a_ctrl acc_ctrl<int>

//==============================================================================
// ACC Specific Constants
//==============================================================================

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

//==============================================================================
// HW Structs
//==============================================================================

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