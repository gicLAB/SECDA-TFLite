#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME VADD_ACC
#define SUBMODULENAME acc_pe

//==============================================================================
// Hardware Constants
//==============================================================================
// OP-Code Struct
// 0000 : 0 = NOP;
// 0001 : 1 = load_X;
// 0010 : 2 = load_Y;
// 0011 : 3 = load_X -> load_Y;
// 0100 : 4 = compute_Z;
// 0101 : 5 = load_X -> compute_Z;
// 0110 : 6 = load_Y -> compute_Z;
// 0111 : 7 = load_X -> load_Y -> compute_Z;
// 1000 : 8 = send_Z;
// 1001 : 9 = load_X -> send_Z;
// 1010 : 10 = load_Y -> send_Z;
// 1011 : 11 = load_X -> load_Y -> send_Z;
// 1100 : 12 = compute_Z -> send_Z;
// 1101 : 13 = load_X -> compute_Z -> send_Z;
// 1110 : 14 = load_Y -> compute_Z -> send_Z;
// 1111 : 15 = load_X -> load_Y -> compute_Z -> send_Z;

//==============================================================================
// Address mapping for the accelerator and DMA
//==============================================================================
#ifdef KRIA
// KRIA
#define acc_ctrl_address 0xA0000000
#define acc_hwc_address 0xA0020000

#define acc_address 0x00A0000000
#define dma_addr0 0xA0010000
#define dma_in0 0x3A000000
#define dma_out0 0x38000000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x0000000037400000
#define DMA_RANGE_END 0x00000000773FFFFF
#define DMA_RANGE_OFFSET 0xC00000 // 1.5MB
#define DMA_RANGE_SIZE 0x40000000 // 1GB

#define DMA_IN_BUF_SIZE 0x3F000000 // 1GB - 16MB
#define DMA_OUT_BUF_SIZE 0x0800000 // 8MB
#define DMA_INP_SIZE 0x100000
#define DMA_WGT_SIZE (DMA_IN_BUF_SIZE - DMA_INP_SIZE)
#else
// Z1
#define acc_ctrl_address 0x43C00000
#define acc_hwc_address 0x43C10000
#define dma_addr0 0x40400000
#define dma_in0 0x18000000
#define dma_out0 0x1C000000

#define DMA_IN_BUF_SIZE 0x0800000 // 8MB
#define DMA_OUT_BUF_SIZE 0x0800000 // 8MB
#define DMA_INP_SIZE 0x100000
#define DMA_WGT_SIZE (DMA_IN_BUF_SIZE - DMA_INP_SIZE)
#define DMA_RANGE_START 0x18000000
#define DMA_RANGE_END 0x1fffffff
#define DMA_RANGE_SIZE 0x8000000
#endif // KRIA

// AXIMM Constants
#ifdef KRIA
#define MM_BL 0x100000 // 1MB
#define in_addr 0x38000000
#define out_addr 0x39000000
#else
// Z1
#define MM_BL 0x100000 // 1MB
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
#define mm_buf_float mm_buffer<float>

#define a_ctrl acc_ctrl<int>
#define h_ctrl hwc_ctrl<int>

//==============================================================================
// ACC Specific Constants
//==============================================================================

// Fixed vector length
const int VEC_LENGTH = 64;

// Buffer sizes
const int X_buffer_size = 4096;
const int Y_buffer_size = 4096;
const int Z_buffer_size = 4096;

// ACC Specific Constants
#define STOPPER -1

#define HWC_Monitor_Count 3
#define CTRL_Reg_Count 1

// Number of PEs
#define ADD_PE_COUNT 0

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

struct _NDATA {
  AXI_TYPE<AXI_DWIDTH> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _NDATA &v) {
    cout << "data: " << v.data << " tlast: " << v.tlast;
    return os;
  }
};

typedef _NDATA ADATA;
#endif

//==============================================================================
// HW Structs
//==============================================================================

struct opcode {
  unsigned int packet;
  bool load_X;
  bool load_Y;
  bool compute_Z;
  bool send_Z;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    load_X = _packet.range(0, 0);
    load_Y = _packet.range(1, 1);
    compute_Z = _packet.range(2, 2);
    send_Z = _packet.range(3, 3);
  }
};

struct code_extension {
  int L;

  code_extension(sc_uint<32> _packetL) {
    L = _packetL;
    ALOG("Time: " << sc_time_stamp());
    ALOG("L: " << L);
  }
};

//==============================================================================
// HW Submodule Construction SIM/HW Structs
//==============================================================================

//==============================================================================

#endif // defined(SYSC) || defined(__SYNTHESIS__)
#endif // ACC_CONFIG_H
