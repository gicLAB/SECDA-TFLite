
#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

// Name of the accelerator
#define ACCNAME VM_RPP_INT8_V14_4

// Pre-Defined Address for Accelerator
#ifdef KRIA
#define acc_address 0xA0000000
#define dma_addr0 0xA0010000
#define dma_addr1 0xA0020000
#define dma_addr2 0xA0030000
#define dma_addr3 0xA0040000
#define DMA_BL 4194304

//////////////////////////////////////////////////////////////////
#define dma_out0 0x38000000
#define dma_out1 0x3A000000
#define dma_out2 0x3C000000
#define dma_out3 0x3E000000
#define dma_in0 0x38800000
#define dma_in1 0x3A800000
#define dma_in2 0x3C800000
#define dma_in3 0x3E800000
// (0x3A000000 - 0x38800000) = 0x1800000 = 24 MB, // for 1 dmabuffer
#define DMA_IN_BUF_SIZE_4 (dma_out1 - dma_in0)
// output buffer size (dma_in0 - dma_out0) = (0x38800000 - 0x38000000) =
// 0x800000 = 8 MB, // for 1 DMA channel

// In DMA Scratch Pad we copy config, input or output data as temporary
// storage, before sending it to accelerator or after receiving it from
// #define DMA_SCRATCH_SIZE 0x00100000   // 1MB, for 1 DMA channel
// 1 MB not successful, 256KB is successful in simulation,
#define DMA_SCRATCH_SIZE_4 0x00040000 // 256KB , for 1 DMA channel

// (0x1800000 - 0x40000) = 0x17C0000 = 24MB - 256KB = 23.75MB, for 1 DMA channel
#define DMA_WGT_SIZE_4 (DMA_IN_BUF_SIZE_4 - DMA_SCRATCH_SIZE_4)
////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// #define dma_in0 0x38000000
// #define dma_in1 0x3A000000
// #define dma_in2 0x3C000000
// #define dma_in3 0x3E000000
// #define dma_out0 0x38800000
// #define dma_out1 0x3A800000
// #define dma_out2 0x3C800000
// #define dma_out3 0x3E800000

// // (0x38800000 - 0x38000000) = 0x800000 = 8MB, for 1 DMA channel
// #define DMA_IN_BUF_SIZE_4 (dma_out0 - dma_in0)
// // output buffer size (dma_in1 - dma_out0) = (0x3A000000 - 0x38800000)
// // = 0x1800000 = 24MB, for 1 DMA channel

// // In DMA Scratch Pad we copy config, input or output data as temporary
// // storage, before sending it to accelerator or after receiving it from
// // accelerator
// // #define DMA_SCRATCH_SIZE 0x00100000   // 1MB, for 1 DMA channel
// // 1 MB not successful, 256KB is successful in simulation,
// #define DMA_SCRATCH_SIZE_4 0x00040000 // 256KB , for 1 DMA channel

// // (0x800000 - 0x40000) = 0x7C0000 = 8MB - 256KB = 7.75MB, for 1 DMA channel
// #define DMA_WGT_SIZE_4 (DMA_IN_BUF_SIZE_4 - DMA_SCRATCH_SIZE_4)
//////////////////////////////////////////////////////////////////

#else // PYNQ

#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
#define DMA_BL 4194304

//////////////////////////////////////////////////////////////////
#define dma_out0 0x16000000
#define dma_out1 0x18000000
#define dma_out2 0x1a000000
#define dma_out3 0x1c000000
#define dma_in0 0x16800000
#define dma_in1 0x18800000
#define dma_in2 0x1a800000
#define dma_in3 0x1c800000

// (0x18000000 - 0x16800000) = 0x1800000 = 24 MB, // for 1 dmabuffer
#define DMA_IN_BUF_SIZE_4 (dma_out1 - dma_in0)
// output buffer size (dma_in0 - dma_out0) = (0x16800000 - 0x16000000) =
// 0x800000 = 8 MB, // for 1 DMA channel

// In DMA Scratch Pad we copy config, input or output data as temporary
// storage, before sending it to accelerator or after receiving it from
// #define DMA_SCRATCH_SIZE 0x00100000   // 1MB, for 1 DMA channel
// 1 MB not successful, 256KB is successful in simulation,
#define DMA_SCRATCH_SIZE_4 0x00040000 // 256KB , for 1 DMA channel

// (0x1800000 - 0x40000) = 0x17C0000 = 24MB - 256KB = 23.75MB, for 1 DMA channel
#define DMA_WGT_SIZE_4 (DMA_IN_BUF_SIZE_4 - DMA_SCRATCH_SIZE_4)
////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// #define dma_in0 0x16000000
// #define dma_in1 0x18000000
// #define dma_in2 0x1a000000
// #define dma_in3 0x1c000000
// #define dma_out0 0x16800000
// #define dma_out1 0x18800000
// #define dma_out2 0x1a800000
// #define dma_out3 0x1c800000

// // (0x16800000 - 0x16000000) = 0x800000 = 8MB, for 1 DMA channel
// #define DMA_IN_BUF_SIZE_4 (dma_out0 - dma_in0)
// // output buffer size (dma_in1 - dma_out0) = (0x18000000 - 0x16800000)
// // = 0x1800000 = 24MB, for 1 DMA channel

// // In DMA Scratch Pad we copy config, input or output data as temporary
// // storage, before sending it to accelerator or after receiving it from
// // #define DMA_SCRATCH_SIZE 0x00100000   // 1MB, for 1 DMA channel
// // 1 MB not successful, 256KB is successful in simulation,
// #define DMA_SCRATCH_SIZE_4 0x00040000 // 256KB , for 1 DMA channel

// // (0x800000 - 0x40000) = 0x7C0000 = 8MB - 256KB = 7.75MB, for 1 DMA channel
// #define DMA_WGT_SIZE_4 (DMA_IN_BUF_SIZE_4 - DMA_SCRATCH_SIZE_4)
//////////////////////////////////////////////////////////////////

#endif

// Accelerator Parameters
// #define VMM_COUNT 1
// #define VMM_COUNT 2
// #define VMM_COUNT 3
#define VMM_COUNT 4 // working
// #define VMM_COUNT 6 // working
// #define VMM_COUNT 8 // working
// #define VMM_COUNT 16 // working
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int

// #define EN_DEPTHTILE
// #define DMA_WGT_PRELOAD

// Buffer Sizes
#ifdef EN_DEPTHTILE
#define DEPTHTILE                                                              \
  2 // when enable EN_DEPTHTILE DEPTHTILE should be >= 2 e.g, 2,4,8
#else
#define DEPTHTILE 1
#endif

// For PYNQ
#define WGT_BUF_LEN 2048
#define INP_BUF_LEN 2048
#define GINP_BUF_LEN 8192

#define WSUMS_BUF_LEN (512 / VMM_COUNT)

#define NO_OF_DATA_CHANNELS 4
#define AXI_BUS_DATA_WIDTH 32
#define ACTIVATION_WIDTH 8
#define INPBLOCK_WIDTH 4 // BLOCK_WIDTH should be multiple of 4
#define WGTBLOCK_WIDTH 4 // BLOCK_WIDTH should be multiple of 4
#define BLOCK_DEPTH 16   // block depth should be multiple of 16
#define ACC_OUT_SIZE 16  // output size of accelerator

#define URAM_DATAWIDTH 64

// Opcodes
#define OPCODE_NOP 0x0
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
#define DLOG(x) std::cout << x << std::endl
// #define DLOG(x)
#define acc_dt sc_int<32>
#define acc_dt_64 sc_int<64>
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
#define DLOG(x)
#define acc_dt sc_int<32>
#define acc_dt_64 sc_int<64>
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
  unsigned int c;
  unsigned int d;
  unsigned int e;
  unsigned int colnoDiv4;
  unsigned int depthDiv4;
  unsigned int minWgtBlockNo;
  unsigned int getExtraWgtBlock;
  unsigned int depth_switch;
  unsigned int wgtBlockArray[VMM_COUNT];
  bool loadWgtArr[VMM_COUNT];

  wgt_packet(sc_fifo_in<DATA> *din) {
    ALOG("WGT_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    c = din->read().data;
    d = din->read().data;
    e = din->read().data;
    colnoDiv4 = a;
    depthDiv4 = b;
    minWgtBlockNo = c;
    getExtraWgtBlock = d;
    depth_switch = e;
    for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
      if (i < getExtraWgtBlock) {
        wgtBlockArray[i] = minWgtBlockNo + 1;
      } else {
        wgtBlockArray[i] = minWgtBlockNo;
      }
    }
    for (int i = 0; i < VMM_COUNT; i++) {
#pragma HLS unroll
      if (wgtBlockArray[i] > 0) {
        loadWgtArr[i] = true;
      } else {
        loadWgtArr[i] = false;
      }
    }
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
    // cout << "idx:" << idx << endl;
    // cout << "read data1: " << (int)(data.range(31, 0))
    //      << "read data2: " << (int)(data.range(63, 32))
    //      << "read data3: " << (int)(data.range(95, 64))
    //      << "read data4: " << (int)(data.range(127, 96)) << endl;
    a1[idx] = data.range(31, 0);
    a2[idx] = data.range(63, 32);
    a3[idx] = data.range(95, 64);
    a4[idx] = data.range(127, 96);
  }
  void unpack_URAM(acc_dt_64 a1[], acc_dt_64 a2[], int idx) {
    a1[idx].range(31, 0) = data.range(31, 0);
    a1[idx].range(63, 32) = data.range(63, 32);
    a2[idx].range(31, 0) = data.range(95, 64);
    a2[idx].range(63, 32) = data.range(127, 96);
  }
} bUF;

struct VMM_vars {
#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> load_inp;
  sc_signal<bool, SC_MANY_WRITERS> load_wgt;
  sc_signal<bool, SC_MANY_WRITERS> load_wsum;
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send_done;
  sc_signal<bool, SC_MANY_WRITERS> ready;
  sc_signal<bool, SC_MANY_WRITERS> vmm_ready;
  sc_signal<bool, SC_MANY_WRITERS> ppu_done;
  sc_signal<bool, SC_MANY_WRITERS> post_ready;
  sc_signal<int, SC_MANY_WRITERS> ra;
  sc_signal<unsigned int, SC_MANY_WRITERS> depth;
  sc_signal<unsigned int, SC_MANY_WRITERS> w_idx;
  sc_signal<unsigned int, SC_MANY_WRITERS> wsum_idx;
  sc_signal<unsigned int, SC_MANY_WRITERS> wgt_len;
  sc_signal<unsigned int, SC_MANY_WRITERS> wsum_len;
  sc_signal<unsigned int, SC_MANY_WRITERS> inp_len;
  sc_signal<unsigned int, SC_MANY_WRITERS> depthLoadWgt;
  sc_signal<unsigned int, SC_MANY_WRITERS> wgt_colno;
#else
  sc_signal<bool> load_inp;
  sc_signal<bool> load_wgt;
  sc_signal<bool> load_wsum;
  sc_signal<bool> compute;
  sc_signal<bool> send_done;
  sc_signal<bool> ready;
  sc_signal<bool> vmm_ready;
  sc_signal<bool> ppu_done;
  sc_signal<bool> post_ready;
  sc_signal<int> ra;
  sc_signal<unsigned int> depth;
  sc_signal<unsigned int> w_idx;
  sc_signal<unsigned int> wsum_idx;
  sc_signal<unsigned int> wgt_len;
  sc_signal<unsigned int> wsum_len;
  sc_signal<unsigned int> inp_len;
  sc_signal<unsigned int> depthLoadWgt;
  sc_signal<unsigned int> wgt_colno;
#endif

  sc_fifo<bUF> wgt_fifo;
  sc_fifo<bUF> inp_fifo;
  sc_fifo<int> post_fifo;
  sc_fifo<bUF> wsum_fifo;
  sc_fifo<bUF> crf_fifo;
  sc_fifo<int> crx_fifo;
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
        load_wsum((std::string("load_wsum") + std::to_string(sid)).c_str()),
        compute((std::string("compute") + std::to_string(sid)).c_str()),
        send_done((std::string("send_done") + std::to_string(sid)).c_str()),
        ready((std::string("ready") + std::to_string(sid)).c_str()),
        vmm_ready((std::string("vmm_ready") + std::to_string(sid)).c_str()),
        ppu_done((std::string("ppu_done") + std::to_string(sid)).c_str()),
        post_ready((std::string("post_ready") + std::to_string(sid)).c_str()),
        depth((std::string("depth") + std::to_string(sid)).c_str()),
        w_idx((std::string("w_idx") + std::to_string(sid)).c_str()),
        wsum_idx((std::string("wsum_idx") + std::to_string(sid)).c_str()),
        wgt_len((std::string("wgt_len") + std::to_string(sid)).c_str()),
        wsum_len((std::string("wsum_len") + std::to_string(sid)).c_str()),
        inp_len((std::string("inp_len") + std::to_string(sid)).c_str()),
        depthLoadWgt(
            (std::string("depthLoadWgt") + std::to_string(sid)).c_str()),
        wgt_colno((std::string("wgt_colno") + std::to_string(sid)).c_str()),
        wgt_fifo(size), inp_fifo(size), post_fifo(size), wsum_fifo(size),
        crf_fifo(size), crx_fifo(size), dout1(size), dout2(size), dout3(size),
        dout4(size),
        computeS((std::string("computeS") + std::to_string(sid)).c_str()),
        postS((std::string("postS") + std::to_string(sid)).c_str()) {}
#else
  VMM_vars(int size)
      : load_inp("load_inp"), load_wgt("load_wgt"), load_wsum("load_wsum"),
        compute("compute"), send_done("send_done"), ready("ready"),
        vmm_ready("vmm_ready"), ppu_done("ppu_done"), post_ready("post_ready"),
        depth("depth"), w_idx("w_idx"), wsum_idx("wsum_idx"),
        wgt_len("wgt_len"), depthLoadWgt("depthLoadWgt"),
        wgt_colno("wgt_colno"), wsum_len("wsum_len"), inp_len("inp_len"),
        wgt_fifo(size), inp_fifo(size), wsum_fifo(size), crf_fifo(size),
        crx_fifo(size), post_fifo(size), dout1(size), dout2(size), dout3(size),
        dout4(size), computeS("computeS"), postS("postS") {
#pragma HLS resource variable = wgt_fifo core = FIFO_SRL
#pragma HLS resource variable = inp_fifo core = FIFO_SRL
#pragma HLS resource variable = wsum_fifo core = FIFO_SRL
#pragma HLS resource variable = crf_fifo core = FIFO_SRL
#pragma HLS resource variable = crx_fifo core = FIFO_SRL
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
