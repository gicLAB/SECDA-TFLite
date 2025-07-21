#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME ADD_ACC
#define SUBMODULENAME add_pe

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
#define a_ctrl acc_ctrl<int>

//==============================================================================
// ACC Specific Constants
//==============================================================================

#define STOPPER -1

// Buffer sizes
#define IN_BUF_LEN 4096
#define WE_BUF_LEN 8192
#define SUMS_BUF_LEN 1024

// Post Processing Parameters
#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648
#define MAX8 127
#define MIN8 -128

#define ADD_PE_COUNT 2

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

//==============================================================================
// HW Submodule Construction SIM/HW Structs
//==============================================================================
struct add_pe_vars {
  // Signal declarations for the hardware submodule
#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> start;
  sc_signal<bool, SC_MANY_WRITERS> done;
  sc_signal<int, SC_MANY_WRITERS> length;
  sc_signal<int, SC_MANY_WRITERS> lshift;
  sc_signal<int, SC_MANY_WRITERS> in1_off;
  sc_signal<int, SC_MANY_WRITERS> in1_sv;
  sc_signal<int, SC_MANY_WRITERS> in1_mul;
  sc_signal<int, SC_MANY_WRITERS> in2_off;
  sc_signal<int, SC_MANY_WRITERS> in2_sv;
  sc_signal<int, SC_MANY_WRITERS> in2_mul;
  sc_signal<int, SC_MANY_WRITERS> out1_off;
  sc_signal<int, SC_MANY_WRITERS> out1_sv;
  sc_signal<int, SC_MANY_WRITERS> out1_mul;
  sc_signal<int, SC_MANY_WRITERS> qa_max;
  sc_signal<int, SC_MANY_WRITERS> qa_min;
#else
  sc_signal<bool> start;
  sc_signal<bool> done;
  sc_signal<int> length;
  sc_signal<int> lshift;
  sc_signal<int> in1_off;
  sc_signal<int> in1_sv;
  sc_signal<int> in1_mul;
  sc_signal<int> in2_off;
  sc_signal<int> in2_sv;
  sc_signal<int> in2_mul;
  sc_signal<int> out1_off;
  sc_signal<int> out1_sv;
  sc_signal<int> out1_mul;
  sc_signal<int> qa_max;
  sc_signal<int> qa_min;
#endif

  // Declare I/O ports and fifos for the hardware submodule
  sc_fifo<int> input_fifo;
  sc_fifo<int> output_fifo;

#ifndef __SYNTHESIS__
  add_pe_vars(int size, int sid)
      : INITSIGPORT(start, sid), INITSIGPORT(done, sid),
        INITSIGPORT(length, sid), INITSIGPORT(lshift, sid),
        INITSIGPORT(in1_off, sid), INITSIGPORT(in1_sv, sid),
        INITSIGPORT(in1_mul, sid), INITSIGPORT(in2_off, sid),
        INITSIGPORT(in2_sv, sid), INITSIGPORT(in2_mul, sid),
        INITSIGPORT(out1_off, sid), INITSIGPORT(out1_sv, sid),
        INITSIGPORT(out1_mul, sid), input_fifo(size), output_fifo(size) {}
#else
  add_pe_vars(int size)
      : start("start"), done("done"), length("length"), lshift("lshift"),
        in1_off("in1_off"), in1_sv("in1_sv"), in1_mul("in1_mul"),
        in2_off("in2_off"), in2_sv("in2_sv"), in2_mul("in2_mul"),
        out1_off("out1_off"), out1_sv("out1_sv"), out1_mul("out1_mul"),
        input_fifo(size), output_fifo(size) {
    // Define any HLS pragma for ports/fifos declared here
    PRAGMA(HLS resource variable = input_fifo core = FIFO_SRL)
    PRAGMA(HLS resource variable = output_fifo core = FIFO_SRL)
  }
#endif
};
//==============================================================================

#endif // defined(SYSC) || defined(__SYNTHESIS__)
#endif // ACC_CONFIG_H