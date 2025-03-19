
#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME MM2IMv2
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int

#define MAX 2147483647
#define MIN -2147483648
#define POS 1073741824
#define NEG -1073741823
#define DIVMAX 2147483648
#define MAX8 127
#define MIN8 -128

#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
#define dma_in0 0x16000000
// #define dma_in1 0x18000000
// #define dma_in2 0x1a000000
// #define dma_in3 0x1c000000
#define dma_out0 0x16800000
// #define dma_out1 0x18800000
// #define dma_out2 0x1a800000
// #define dma_out3 0x1c800000
// #define dma_in0 0x18000000
// #define dma_out0 0x18800000
#define DMA_BL 4194304

// HERE //TCONV SYNTH v2_1/v2_4  Works
#define PE_COUNT 8
#define UF 16
#define SUP_STRIDE 2
#define SUP_IHW 11
#define SUP_KS 7
#define SUP_DEPTH 256
#define SUP_IR 8
#define SUP_OHW ((SUP_IHW - 1) * SUP_STRIDE) + SUP_KS // 27
// #define SUP_OHW (SUP_IHW * SUP_STRIDE)
#define INP_BUF_LEN 2048
// #define WGT_BUF_LEN (SUP_DEPTH * SUP_KS * SUP_KS * PE_COUNT) / UF
#define WGT_BUF_LEN 2048 * 4
#define PE_WGTCOLBUF_SIZE ((SUP_KS * SUP_KS * SUP_DEPTH) / UF) // 784
// #define PE_WGTCOLBUF_SIZE 784
#define PE_WGTCOLSUMBUF_SIZE (SUP_KS * SUP_KS) // 49
// #define PE_WGTCOLSUMBUF_SIZE 64
#define PE_INPROWBUF_SIZE (SUP_DEPTH / UF) // 16
// #define PE_INPROWBUF_SIZE 16
#define PE_OUTBUF_SIZE (SUP_IR * SUP_KS * SUP_KS) // 392
// #define PE_OUTBUF_SIZE 512
#define PE_POUTDEXBUF_SIZE (SUP_KS * SUP_KS) // 49
// #define PE_POUTDEXBUF_SIZE 64
// #define PE_ACC_BUF_SIZE (SUP_OHW * SUP_OHW) // 484
#define PE_ACC_BUF_SIZE 1024
#define G_WGTSUMBUF_SIZE (SUP_KS * SUP_KS * PE_COUNT) // 392
// #define G_WGTSUMBUF_SIZE 512
// TO HERE



// // HERE //DCGAN  v2_3
// #define SUP_STRIDE 2
// #define SUP_IHW 16
// #define SUP_KS 5
// #define SUP_DEPTH 1024

// #define SUP_IR 8
// #define SUP_OHW (SUP_IHW * SUP_STRIDE)

// // #define SUP_OHW ((SUP_IHW - 1) * SUP_STRIDE) + SUP_KS // 27

// // Number of PEs
// #define PE_COUNT 8
// #define UF 16

// #define INP_BUF_LEN 2048
// #define WGT_BUF_LEN (SUP_DEPTH * SUP_KS * SUP_KS * PE_COUNT) / UF

// // needs to support ks * ks * depth / UF
// #define PE_WGTCOLBUF_SIZE ((SUP_KS * SUP_KS * SUP_DEPTH) / UF) // 1936
// // #define PE_WGTCOLBUF_SIZE 512

// // wgt_col_sum needs to support ks * ks
// #define PE_WGTCOLSUMBUF_SIZE (SUP_KS * SUP_KS) // 49
// // #define PE_WGTCOLSUMBUF_SIZE 64

// // inp_row_buf needs to support depth / UF
// #define PE_INPROWBUF_SIZE (SUP_DEPTH / UF) // 16
// // #define PE_INPROWBUF_SIZE 16

// // support input_rows * ks * ks gemm outputs
// #define PE_OUTBUF_SIZE (SUP_IR * SUP_KS * SUP_KS) // 968
// // #define PE_OUTBUF_SIZE 1024

// // max value is ks * ks
// #define PE_POUTDEXBUF_SIZE (SUP_KS * SUP_KS) // 121
// // #define PE_POUTDEXBUF_SIZE 64

// // Max number of MM2IM outputs storable per PE, should allow OH * OW
// // #define PE_ACC_BUF_SIZE (SUP_OHW * SUP_OHW) // 729
// #define PE_ACC_BUF_SIZE (1024) // 729
// // #define PE_ACC_BUF_SIZE 256

// // Needs to support ks * ks * PE_COUNT
// #define G_WGTSUMBUF_SIZE (SUP_KS * SUP_KS * PE_COUNT) // 968
// // #define G_WGTSUMBUF_SIZE 200

// // TO HERE

#if defined(SYSC) || defined(__SYNTHESIS__)

#include <systemc.h>
#ifndef __SYNTHESIS__
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_integrator/sysc_types.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
// typedef sc_int<8>  acc_dt;
#define acc_dt sc_int<8>
#define DWAIT(x) wait(x)
#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif
#else

typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;

template <typename T, unsigned int W>
struct sbram {
  T data[W];
  int idx;
#ifndef __SYNTHESIS__
  int size;
  int access_count;
#endif
  sbram() {
#ifndef __SYNTHESIS__
    size = W;
    access_count = 0;
#endif
  }
  T &operator[](int i) {
    idx = i;
    return data[i];
  }
  int &operator=(int val) { data[idx] = val; }
  void write(int i, T val) { data[i] = val; }
  T read(int i) { return data[i]; }
};

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

// typedef sc_int<8> acc_dt;
#define acc_dt sc_int<8>
#define DWAIT(x)
#define ALOG(x)
#endif
#define ATOG(x)

struct opcode {
  unsigned int packet;
  bool load_con;
  bool load_wgt;
  bool load_inp;
  bool load_map;
  bool schedule;
  bool load_col_map;
  bool store;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    load_con = _packet.range(0, 0);
    load_wgt = _packet.range(1, 1);
    load_inp = _packet.range(2, 2);
    load_map = _packet.range(3, 3);
    schedule = _packet.range(4, 4);
    load_col_map = _packet.range(5, 5);
    store = _packet.range(6, 6);
  }
};

struct wgt_packet {
  unsigned int a;
  unsigned int b;
  unsigned int wgt_rows;
  unsigned int wgt_depth;

  wgt_packet(sc_fifo_in<DATA> *din) {
    ALOG("WGT_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    wgt_rows = a;
    wgt_depth = b;
  }
};

struct inp_packet {
  unsigned int a;
  unsigned int b;
  unsigned int c;

  unsigned int inp_rows;
  unsigned int inp_depth;
  unsigned int srow;

  inp_packet(sc_fifo_in<DATA> *din) {
    ALOG("INP_PACKET");
    ALOG("Time: " << sc_time_stamp());
    a = din->read().data;
    b = din->read().data;
    c = din->read().data;
    inp_rows = a;
    inp_depth = b;
    srow = c;
  }
};

typedef struct _DATA_PACKED {
  sc_uint<32> data;
  int c;
  _DATA_PACKED() { c = 0; }

  void insert(sc_uint<8> _data) {
    data.range(c + 7, c) = _data;
    c += 8;
    c = c % 32;
  }

} DATA_PACKED;

typedef struct byteToUF {
  sc_bigint<8 * UF> data;

  void insert(acc_dt array[][UF], int index) {
    for (int i = 0; i < UF; i++) {
#pragma HLS unroll
      data.range(((i + 1) * 8) - 1, i * 8) = array[index][i];
    }
  }

  void insert(acc_dt array[UF]) {
    for (int i = 0; i < UF; i++) {
#pragma HLS unroll
      data.range(((i + 1) * 8) - 1, i * 8) = array[i];
    }
  }

  void insert(sc_uint<32> _data, int i) {
    data.range(((i + 1) * 8) - 1, i * 8) = _data.range(7, 0);
    data.range(((i + 2) * 8) - 1, (i + 1) * 8) = _data.range(15, 8);
    data.range(((i + 3) * 8) - 1, (i + 2) * 8) = _data.range(23, 16);
    data.range(((i + 4) * 8) - 1, (i + 3) * 8) = _data.range(31, 24);
  }

  void retreive(acc_dt array[][UF], int index) {
    for (int i = 0; i < UF; i++) {
#pragma HLS unroll
      array[index][i] = data.range(((i + 1) * 8) - 1, i * 8).to_int();
    }
  }

  void operator=(byteToUF _data) { data = _data.data; }
  inline friend ostream &operator<<(ostream &os, const byteToUF &v) {
    cout << "data&colon; " << v.data;
    return os;
  }
} bUF;

// struct sc_out_sig {
//   sc_out<int> oS;
//   sc_signal<int> iS;
//   void write(int x) {
//     oS.write(x);
//     iS.write(x);
//   }
//   int read() { return iS.read(); }
//   void operator=(int x) { write(x); }
//   void bind(sc_signal<int> &sig) { oS.bind(sig); }
//   void operator()(sc_signal<int> &sig) { bind(sig); }
//   void bind(sc_out<int> &sig) { oS.bind(sig); }
//   void operator()(sc_out<int> &sig) { bind(sig); }
// };

struct PE_vars {

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> online;
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> reset_compute;
  sc_signal<int, SC_MANY_WRITERS> start_addr_p;
  sc_signal<int, SC_MANY_WRITERS> send_len_p;
  sc_signal<int, SC_MANY_WRITERS> bias_data;
  sc_signal<int, SC_MANY_WRITERS> crf_data;
  sc_signal<int, SC_MANY_WRITERS> crx_data;
  sc_signal<int, SC_MANY_WRITERS> ra_data;
  sc_signal<bool, SC_MANY_WRITERS> send;
  sc_signal<bool, SC_MANY_WRITERS> out;
  sc_signal<int, SC_MANY_WRITERS> cols_per_filter;
  sc_signal<int, SC_MANY_WRITERS> depth;
  sc_signal<bool, SC_MANY_WRITERS> compute_done;
  sc_signal<bool, SC_MANY_WRITERS> wgt_loaded;
  sc_signal<bool, SC_MANY_WRITERS> out_done;
  sc_signal<bool, SC_MANY_WRITERS> send_done;

  sc_signal<bool, SC_MANY_WRITERS> process_cal;
  sc_signal<bool, SC_MANY_WRITERS> process_cal_done;

  sc_signal<int, SC_MANY_WRITERS> oh;
  sc_signal<int, SC_MANY_WRITERS> ow;
  sc_signal<int, SC_MANY_WRITERS> kernel_size;
  sc_signal<int, SC_MANY_WRITERS> stride_x;
  sc_signal<int, SC_MANY_WRITERS> stride_y;
  sc_signal<int, SC_MANY_WRITERS> pt;
  sc_signal<int, SC_MANY_WRITERS> pl;
  sc_signal<int, SC_MANY_WRITERS> width_col;
  sc_signal<int, SC_MANY_WRITERS> crow;
  sc_signal<int, SC_MANY_WRITERS> num_rows;
#else
  sc_signal<bool> online;
  sc_signal<bool> compute;
  sc_signal<bool> reset_compute;
  sc_signal<int> start_addr_p;
  sc_signal<int> send_len_p;
  sc_signal<int> bias_data;
  sc_signal<int> crf_data;
  sc_signal<int> crx_data;
  sc_signal<int> ra_data;
  sc_signal<bool> send;
  sc_signal<bool> out;
  sc_signal<int> cols_per_filter;
  sc_signal<int> depth;
  sc_signal<bool> compute_done;
  sc_signal<bool> wgt_loaded;
  sc_signal<bool> out_done;
  sc_signal<bool> send_done;

  sc_signal<bool> process_cal;
  sc_signal<bool> process_cal_done;

  sc_signal<int> oh;
  sc_signal<int> ow;
  sc_signal<int> kernel_size;
  sc_signal<int> stride_x;
  sc_signal<int> stride_y;
  sc_signal<int> pt;
  sc_signal<int> pl;
  sc_signal<int> width_col;
  sc_signal<int> crow;
  sc_signal<int> num_rows;

#endif

  sc_fifo<int> wgt_sum_fifo;
  sc_fifo<bUF> wgt_fifo;
  sc_fifo<bUF> inp_fifo;
  sc_fifo<DATA> out_fifo;
  sc_fifo<int> temp_fifo;

  sc_fifo<DATA> col_indices_fifo;
  sc_fifo<DATA> out_indices_fifo;

  sc_out<int> computeS;
  sc_out<int> sendS;

#ifndef __SYNTHESIS__
  PE_vars(int size, int sid)
      : online((std::string("online") + std::to_string(sid)).c_str()),
        compute((std::string("compute") + std::to_string(sid)).c_str()),
        reset_compute(
            (std::string("reset_compute") + std::to_string(sid)).c_str()),
        start_addr_p(
            (std::string("start_addr_p") + std::to_string(sid)).c_str()),
        send_len_p((std::string("send_len_p") + std::to_string(sid)).c_str()),
        bias_data((std::string("bias_data") + std::to_string(sid)).c_str()),
        crf_data((std::string("crf_data") + std::to_string(sid)).c_str()),
        crx_data((std::string("crx_data") + std::to_string(sid)).c_str()),
        ra_data((std::string("ra_data") + std::to_string(sid)).c_str()),
        send((std::string("send") + std::to_string(sid)).c_str()),
        out((std::string("out") + std::to_string(sid)).c_str()),
        cols_per_filter(
            (std::string("cols_per_filter") + std::to_string(sid)).c_str()),
        depth((std::string("depth") + std::to_string(sid)).c_str()),
        compute_done(
            (std::string("compute_done") + std::to_string(sid)).c_str()),
        wgt_loaded((std::string("wgt_loaded") + std::to_string(sid)).c_str()),
        out_done((std::string("out_done") + std::to_string(sid)).c_str()),
        send_done((std::string("send_done") + std::to_string(sid)).c_str()),
        process_cal((std::string("process_cal") + std::to_string(sid)).c_str()),
        process_cal_done(
            (std::string("process_cal_done") + std::to_string(sid)).c_str()),
        oh((std::string("oh") + std::to_string(sid)).c_str()),
        ow((std::string("ow") + std::to_string(sid)).c_str()),
        kernel_size((std::string("kernel_size") + std::to_string(sid)).c_str()),
        stride_x((std::string("stride_x") + std::to_string(sid)).c_str()),
        stride_y((std::string("stride_y") + std::to_string(sid)).c_str()),
        pt((std::string("pt") + std::to_string(sid)).c_str()),
        pl((std::string("pl") + std::to_string(sid)).c_str()),
        width_col((std::string("width_col") + std::to_string(sid)).c_str()),
        crow((std::string("crow") + std::to_string(sid)).c_str()),
        num_rows((std::string("num_rows") + std::to_string(sid)).c_str()),
        wgt_sum_fifo(size), wgt_fifo(size), inp_fifo(size), out_fifo(size),
        // temp_fifo(size), col_indices_fifo(SUP_KS * SUP_KS),
        // out_indices_fifo(SUP_KS * SUP_KS),
        temp_fifo(size), col_indices_fifo(SUP_KS * SUP_KS* 10),
        out_indices_fifo(SUP_KS * SUP_KS * 10),
        computeS((std::string("computeS") + std::to_string(sid)).c_str()),
        sendS((std::string("sendS") + std::to_string(sid)).c_str()) {}
#else
  PE_vars(int size)
      : online("online"), compute("compute"), reset_compute("reset_compute"),
        start_addr_p("start_addr_p"), send_len_p("send_len_p"),
        bias_data("bias_data"), crf_data("crf_data"), crx_data("crx_data"),
        ra_data("ra_data"), send("send"), out("out"),
        cols_per_filter("cols_per_filter"), depth("depth"),
        compute_done("compute_done"), wgt_loaded("wgt_loaded"),
        out_done("out_done"), send_done("send_done"),
        process_cal("process_cal"), process_cal_done("process_cal_done"),
        oh("oh"), ow("ow"), kernel_size("kernel_size"), stride_x("stride_x"),
        stride_y("stride_y"), pt("pt"), pl("pl"), width_col("width_col"),
        crow("crow"), num_rows("num_rows"), wgt_sum_fifo(size), wgt_fifo(size),
        inp_fifo(size), out_fifo(size), temp_fifo(size),
        col_indices_fifo(SUP_KS * SUP_KS), out_indices_fifo(SUP_KS * SUP_KS),
        computeS("computeS"), sendS("sendS") {
#pragma HLS resource variable = wgt_fifo core = FIFO_SRL
#pragma HLS resource variable = inp_fifo core = FIFO_SRL
#pragma HLS resource variable = out_fifo core = FIFO_SRL
#pragma HLS resource variable = col_indices_fifo core = FIFO_SRL
#pragma HLS resource variable = out_indices_fifo core = FIFO_SRL
  }
#endif
};

#endif // defined(SYSC) || defined(__SYNTHESIS__)

#endif // ACC_CONFIG_H