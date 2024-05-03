
#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

// Name of the accelerator
#define ACCNAME SA_INT8_V3_0

// Pre-Defined Address for Accelerator
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

// Accelerator Parameters
#define SA_SIZE_X 16
#define SA_SIZE_Y 16
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int

// Opcodes
#define OPCODE_LOAD_WGT 0x1
#define OPCODE_LOAD_INP 0x2
#define OPCODE_COMPUTE 0x4
#define OPCODE_CONFIG 0x8

// Buffer Sizes
#define IN_BUF_LEN 4096
#define WE_BUF_LEN 8192
#define SUMS_BUF_LEN 1024

#if defined(SYSC) || defined(__SYNTHESIS__)

#include <systemc.h>
#ifndef __SYNTHESIS__
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_integrator/sysc_types.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#define DWAIT(x) wait(x)
// #define ALOG(x) std::cout << x << std::endl
#define ALOG(x)
#define acc_dt sc_int<32>
#else
typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;
  void operator=(_DATA _data) {
    data = _data.data;
    tlast = _data.tlast;
  }
  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
  void pack(ACC_DTYPE<8> a1, ACC_DTYPE<8> a2, ACC_DTYPE<8> a3,
            ACC_DTYPE<8> a4) {
    data.range(7, 0) = a1;
    data.range(15, 8) = a2;
    data.range(23, 16) = a3;
    data.range(31, 24) = a4;
  }
} DATA;

struct sc_out_sig {
  sc_out<int> oS;
  sc_signal<int> iS;
  void write(int x) {
    oS.write(x);
    iS.write(x);
  }
  int read() { return iS.read(); }
  void operator=(int x) { write(x); }
  void bind(sc_signal<int> &sig) { oS.bind(sig); }
  void operator()(sc_signal<int> &sig) { bind(sig); }
  void bind(sc_out<int> &sig) { oS.bind(sig); }
  void operator()(sc_out<int> &sig) { bind(sig); }
};
#define DWAIT(x)
#define ALOG(x)
#define acc_dt sc_int<32>
#endif

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
  inp_packet(sc_fifo_in<DATA> *din) {
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
  wgt_packet(sc_fifo_in<DATA> *din) {
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
  compute_packet(sc_fifo_in<DATA> *din) {
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
  config_packet(sc_fifo_in<DATA> *din) {
    ALOG("CON_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    depth = a;
    ra = b;
  }
};


#endif // SYSC || __SYNTHESIS__

#endif // ACC_CONFIG_H
