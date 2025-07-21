
#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

// Name of the accelerator
#define ACCNAME SA_INT8_V4_0

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
#define acc_ctrl_address 0x00A0000000
#define acc_hwc_address 0x00A0010000

#define dma_addr0 0x00A0010000
#define dma_addr1 0x00A0020000
#define dma_addr2 0x00A0030000
#define dma_addr3 0x00A0040000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x0000000037400000
#define DMA_RANGE_END 0x00000000773FFFFF
#define DMA_RANGE_OFFSET 0xC00000          // 1.5MB
#define DMA_RANGE_SIZE 0x0000000040000000  // 1GB
#define DMA_IN_BUF_SIZE 0x20000000         // 32MB
#define DMA_OUT_BUF_SIZE 0x20000000        // 32MB

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
#endif  // KRIA

// AXIMM Constants
#ifdef KRIA
#define MM_BL 0x100000  // 1MB
#define in_addr 0x38000000
#define out_addr 0x39000000
#else
// Z1
#define MM_BL 0x100000  // 1MB
#define in_addr 0x18000000
#define out_addr 0x19000000
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
#define a_ctrl acc_ctrl<int>

//==============================================================================
// ACC Specific Constants
//==============================================================================

// Accelerator Parameters
#define SA_SIZE_X 16
#define SA_SIZE_Y 16

// Opcodes
#define OPCODE_LOAD_WGT 0x1
#define OPCODE_LOAD_INP 0x2
#define OPCODE_COMPUTE 0x4
#define OPCODE_CONFIG 0x8

// Buffer Sizes
#define IN_BUF_LEN 4096
#define WE_BUF_LEN 8192
#define SUMS_BUF_LEN 1024

#define PROD_DATA_WIDTH_NORM 32
#define PROD_DATA_WIDTH_MSQ 12
#define PROD_DATA_WIDTH_APOT 14
#define PROD_DATA_WIDTH_QKERAS 15
#define PROD_DATA_WIDTH_8x4 12

#define NORM
// #define QKERAS
// #define MSQ
// #define APOT

// Change as needed
#if defined(QKERAS)
#define PROD_DATA_WIDTH PROD_DATA_WIDTH_QKERAS
#elif defined(MSQ)
#define PROD_DATA_WIDTH PROD_DATA_WIDTH_MSQ
#elif defined(APOT)
#define PROD_DATA_WIDTH PROD_DATA_WIDTH_APOT
#else
#define PROD_DATA_WIDTH PROD_DATA_WIDTH_8x4
#endif

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
#else  // !VERBOSE_ACC
#define ALOG(x)
#endif

typedef _BDATA<AXI_DWIDTH, AXI_TYPE> ADATA;

#else  // __SYNTHESIS__
#include "sysc_types.h"
#define ALOG(x)

struct _NDATA {
  AXI_TYPE<AXI_DWIDTH> data;
  bool tlast;
  inline friend ostream& operator<<(ostream& os, const _NDATA& v) {
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

// PPU Scalers
#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648
#define MAX8 127
#define MIN8 -128

struct opcode {
  unsigned int packet;
  bool load_wgt;
  bool load_inp;
  bool compute;
  bool config;
  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    load_wgt = _packet.range(0, 0);
    load_inp = _packet.range(1, 1);
    compute = _packet.range(2, 2);
    config = _packet.range(3, 3);
  }
};

struct inp_packet {
  unsigned int a;
  unsigned int b;
  unsigned int inp_size;
  unsigned int inp_sum_size;
  inp_packet(sc_fifo_in<ADATA>* din) {
    ALOG("INP_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    inp_size = a;
    inp_sum_size = b;
  }
};

struct wgt_packet {
  unsigned int a;
  unsigned int b;
  unsigned int wgt_size;
  unsigned int wgt_sum_size;
  wgt_packet(sc_fifo_in<ADATA>* din) {
    ALOG("WGT_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    wgt_size = a;
    wgt_sum_size = b;
  }
};

struct compute_packet {
  unsigned int a;
  unsigned int b;
  unsigned int c;
  unsigned int inp_block;
  unsigned int wgt_block;
  compute_packet(sc_fifo_in<ADATA>* din) {
    ALOG("COM_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    inp_block = a;
    wgt_block = b;
  }
};

struct config_packet {
  unsigned int a;
  unsigned int b;
  unsigned int depth;
  unsigned int ra;
  config_packet(sc_fifo_in<ADATA>* din) {
    ALOG("CON_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    depth = a;
    ra = b;
  }
};

//==============================================================================
// HW Submodule Construction SIM/HW Structs
//==============================================================================

#endif  // SYSC || __SYNTHESIS__

#endif  // ACC_CONFIG_H
