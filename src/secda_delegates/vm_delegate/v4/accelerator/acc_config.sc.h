
#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

// Name of the accelerator
#define ACCNAME VM_INT8_V4_0

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
#define DMA_RANGE_OFFSET 0xC00000       // 1.5MB
#define DMA_RANGE_SIZE 0x0000000040000000 // 1GB
#define DMA_IN_BUF_SIZE 0x20000000        // 32MB
#define DMA_OUT_BUF_SIZE 0x20000000       // 32MB

// #define dma_in0 DMA_RANGE_START + DMA_RANGE_OFFSET
// #define dma_in1 dma_in0 + DMA_IN_BUF_SIZE
// #define dma_in2 dma_in1 + DMA_IN_BUF_SIZE
// #define dma_in3 dma_in2 + DMA_IN_BUF_SIZE

// #define dma_out0 dma_in3 + DMA_IN_BUF_SIZE
// #define dma_out1 dma_out0 + DMA_OUT_BUF_SIZE
// #define dma_out2 dma_out1 + DMA_OUT_BUF_SIZE
// #define dma_out3 dma_out2 + DMA_OUT_BUF_SIZE


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
#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
// #define dma_in0 0x16000000
// #define dma_in1 0x18000000
// #define dma_in2 0x1a000000
// #define dma_in3 0x1c000000
// #define dma_out0 0x16800000
// #define dma_out1 0x18800000
// #define dma_out2 0x1a800000
// #define dma_out3 0x1c800000

#define dma_in0 0x18000000
#define dma_in1 0x1a000000
#define dma_in2 0x1c000000
#define dma_in3 0x1e000000
#define dma_out0 0x18800000
#define dma_out1 0x1a800000
#define dma_out2 0x1c800000
#define dma_out3 0x1e800000

// #define dma_in0 0x18800000
// #define dma_in1 0x1a800000
// #define dma_in2 0x1c800000
// #define dma_in3 0x1e800000
// #define dma_out0 0x19000000
// #define dma_out1 0x1b000000
// #define dma_out2 0x1d000000
// #define dma_out3 0x1f000000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x18000000
#define DMA_RANGE_END 0x1fffffff
#define DMA_RANGE_SIZE 0x8000000
#endif

// Accelerator Parameters
// #define VMM_COUNT 1
// #define VMM_COUNT 2
// #define VMM_COUNT 3
#define VMM_COUNT 4
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int

// Buffer Sizes
#define WGT_BUF_LEN 2048
#define INP_BUF_LEN 2048
#define GINP_BUF_LEN 8192
#define WSUMS_BUF_LEN 512
#define ISUMS_BUF_LEN 512
#define SUMS_BUF_LEN 512

// Opcodes
#define OPCODE_LOAD_WGT 0x1
#define OPCODE_LOAD_INP 0x2
#define OPCODE_COMPUTE 0x4
#define OPCODE_CONFIG 0x8

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

typedef struct packed8x4 {
  sc_uint<32> data;
  void operator=(packed8x4 _data) { data = _data.data; }
  inline friend ostream &operator<<(ostream &os, const packed8x4 &v) {
    cout << "data&colon; " << v.data;
    return os;
  }
  void pack(ACC_DTYPE<8> a1, ACC_DTYPE<8> a2, ACC_DTYPE<8> a3,
            ACC_DTYPE<8> a4) {
    data.range(7, 0) = a1;
    data.range(15, 8) = a2;
    data.range(23, 16) = a3;
    data.range(31, 24) = a4;
  }
} p8x4;

typedef struct byteToUF {
  sc_bigint<32 * 4> data;
  void operator=(byteToUF _data) { data = _data.data; }
  void operator=(sc_bigint<32 * 4> _data) { data = _data.range(127, 0); }
  byteToUF() {}
  byteToUF(sc_bigint<32 * 4> _data) { data = _data.range(127, 0); }
  inline friend ostream &operator<<(ostream &os, const byteToUF &v) {
    cout << "data&colon; " << v.data;
    return os;
  }
  void unpack(acc_dt a1[], acc_dt a2[], acc_dt a3[], acc_dt a4[], int idx) {
    a1[idx] = data.range(31, 0);
    a2[idx] = data.range(63, 32);
    a3[idx] = data.range(95, 64);
    a4[idx] = data.range(127, 96);
  }
} bUF;

struct VMM_vars {
#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> load_inp;
  sc_signal<bool, SC_MANY_WRITERS> load_wgt;
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send_done;
  sc_signal<bool, SC_MANY_WRITERS> ready;
  sc_signal<bool, SC_MANY_WRITERS> vmm_ready;
  sc_signal<bool, SC_MANY_WRITERS> ppu_done;
  sc_signal<int, SC_MANY_WRITERS> ra;
  sc_signal<unsigned int, SC_MANY_WRITERS> depth;
  sc_signal<unsigned int, SC_MANY_WRITERS> w_idx;
  sc_signal<unsigned int, SC_MANY_WRITERS> wgt_len;
  sc_signal<unsigned int, SC_MANY_WRITERS> inp_len;
#else
  sc_signal<bool> load_inp;
  sc_signal<bool> load_wgt;
  sc_signal<bool> compute;
  sc_signal<bool> send_done;
  sc_signal<bool> ready;
  sc_signal<bool> vmm_ready;
  sc_signal<bool> ppu_done;
  sc_signal<int> ra;
  sc_signal<unsigned int> depth;
  sc_signal<unsigned int> w_idx;
  sc_signal<unsigned int> wgt_len;
  sc_signal<unsigned int> inp_len;
#endif

  sc_fifo<bUF> wgt_fifo;
  sc_fifo<bUF> inp_fifo;
  sc_fifo<int> post_fifo;
  sc_fifo<DATA> dout1;
  sc_fifo<DATA> dout2;
  sc_fifo<DATA> dout3;
  sc_fifo<DATA> dout4;
  sc_out<int> computeS;
  sc_out<int> postS;

#ifndef __SYNTHESIS__
  VMM_vars(int size, int sid)
      : load_inp((std::string("load_inp") + std::to_string(sid)).c_str()),
        load_wgt((std::string("load_wgt") + std::to_string(sid)).c_str()),
        compute((std::string("compute") + std::to_string(sid)).c_str()),
        send_done((std::string("send_done") + std::to_string(sid)).c_str()),
        ready((std::string("ready") + std::to_string(sid)).c_str()),
        vmm_ready((std::string("vmm_ready") + std::to_string(sid)).c_str()),
        ppu_done((std::string("ppu_done") + std::to_string(sid)).c_str()),
        depth((std::string("depth") + std::to_string(sid)).c_str()),
        w_idx((std::string("w_idx") + std::to_string(sid)).c_str()),
        wgt_len((std::string("wgt_len") + std::to_string(sid)).c_str()),
        inp_len((std::string("inp_len") + std::to_string(sid)).c_str()),
        wgt_fifo(size), inp_fifo(size), post_fifo(size), dout1(size),
        dout2(size), dout3(size), dout4(size),
        computeS((std::string("computeS") + std::to_string(sid)).c_str()),
        postS((std::string("postS") + std::to_string(sid)).c_str()) {}
#else
  VMM_vars(int size)
      : load_inp("load_inp"), load_wgt("load_wgt"), compute("compute"),
        send_done("send_done"), ready("ready"), vmm_ready("vmm_ready"),
        ppu_done("ppu_done"), depth("depth"), w_idx("w_idx"),
        wgt_len("wgt_len"), inp_len("inp_len"), wgt_fifo(size), inp_fifo(size),
        post_fifo(size), dout1(size), dout2(size), dout3(size), dout4(size),
        computeS("computeS"), postS("postS") {
#pragma HLS resource variable = wgt_fifo core = FIFO_SRL
#pragma HLS resource variable = inp_fifo core = FIFO_SRL
#pragma HLS resource variable = post_fifo core = FIFO_SRL
#pragma HLS resource variable = dout1 core = FIFO_SRL
#pragma HLS resource variable = dout2 core = FIFO_SRL
#pragma HLS resource variable = dout3 core = FIFO_SRL
#pragma HLS resource variable = dout4 core = FIFO_SRL
  }
#endif
};

#endif // SYSC || __SYNTHESIS__

#endif // ACC_CONFIG_H
