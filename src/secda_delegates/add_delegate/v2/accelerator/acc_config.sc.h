#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME TOYADD_ACC
#define SUBMODULENAME add_pe

//==============================================================================
// Hardware Constants
//==============================================================================
// Define any Hardware specific constants for the accelerator
// These constants will be accessible in the driver
// These constants will be used to generate the hardware

// Address mapping for the accelerator and DMA
#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_in0 0x16000000
#define dma_out0 0x16800000
#define DMA_BL 4194304

// Data types
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define AXI_DWIDTH 32
#define AXI_TYPE sc_uint

// ACC Specific Constants
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
#endif

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
#pragma HLS resource variable = input_fifo core = FIFO_SRL
#pragma HLS resource variable = output_fifo core = FIFO_SRL
  }
#endif
};
//==============================================================================

#endif // defined(SYSC) || defined(__SYNTHESIS__)
#endif // ACC_CONFIG_H