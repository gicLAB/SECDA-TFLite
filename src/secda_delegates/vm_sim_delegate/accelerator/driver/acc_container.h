#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include "../acc.h"
#include "systemc_binding.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include <vector>

// Used for storing current GEMM info
struct gemm_details {
  int layer = 0;
  int layer_weight_tile = 0;
  int layer_input_tile = 0;
  int layer_print = -1;
  int layer_ww = -1;
  int layer_iw = -1;
  bool profile = false;
};

// Used for tracking output locations
struct store_params {
  int *dst;
  int dcs;
  int rows;
  int cols;
  int rcols;
  int rrows;
};

struct acc_container {
  // Gives driver access to SystemC modules + profiler
  ACCNAME *acc;
  Profile *profile;
  struct multi_dma *mdma;

  // Temporary Weight non-MMapped Padded Buffers
  int *wb_0;
  int *wb_1;
  int *wb_2;
  int *wb_3;

  // Temporary Input non-MMapped Padded Buffers
  int *inb_0;
  int *inb_1;
  int *inb_2;
  int *inb_3;
  int in_id = 0;

  // Driver variables
  struct store_params st_params;
  int w_c = 0;

  // Output Pipeline Metadata
  std::vector<int> wt_sum1;
  std::vector<int> wt_sum2;
  std::vector<int> wt_sum3;
  std::vector<int> wt_sum4;
  int *in_sum1;
  int *in_sum2;
  int *in_sum3;
  int *in_sum4;
  int *bias;
  std::vector<int> crf;
  std::vector<int8_t> crx;
  int ra;
  int rhs_offset = 0;
  int lhs_offset = 0;

  int rows = 0;
  int cols = 0;
  int depth = 0;
  int8_t *dst;

  // GEMM Info variable
  struct gemm_details t;
  int layer = 0;
  bool use_sim = false;

  void clear_traces() {
    if (!use_sim) {
      ofstream file;
      std::string filename =
          ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
      file.open(filename, std::ios_base::trunc);
      file.close();
    }
  }

  // TODO generate read per byte
  template <typename T>
  void massign(T *dst, T *src, int d_dex, int s_dex, T value) {
    if (!use_sim) {
      cout << "dst_addr: " << (void *)&dst[d_dex] << endl;
      auto dst_addr = (void *)(&dst[d_dex]);
      auto src_addr = (void *)(&src[s_dex]);
      // truncate hex address to 32 bits
      dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
      src_addr = (void *)((uint64_t)src_addr & 0xffffffff);
      // create memory trace for ramulator and write to file
      ofstream file;
      std::string filename =
          ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
      file.open(filename, std::ios_base::app);
      file << std::hex << (src_addr++) << " R" << endl;
      file << std::hex << (dst_addr++) << " W" << endl;
      file.close();
    }
    // dst[d_dex] = value;
  }

  template <typename D, typename S>
  void massign(D *dst, S *src, int d_dex, int s_dex) {
    auto dst_addr = (void *)(&dst[d_dex]);
    auto src_addr = (void *)(&src[s_dex]);
    // truncate hex address to 32 bits
    dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
    src_addr = (void *)((uint64_t)src_addr & 0xffffffff);
    // create memory trace for ramulator and write to file
    ofstream file;
    std::string filename =
        ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
    file.open(filename, std::ios_base::app);
    // file << (uint64_t) (src_addr++) << " R" << endl;
    // file << (uint64_t) (dst_addr++) << " W" << endl;

    file << " 3 " << (uint64_t)(src_addr++) << " " << (uint64_t)(dst_addr++)
         << endl;

    // file << std::hex << (src_addr++) << " R" << endl;
    // file << std::hex << (dst_addr++) << " W" << endl;
    file.close();
  }

  template <typename D, typename S>
  void massign(D *dst, S *src, int d_dex, int s_dex, int cpu_ins) {
    auto dst_addr = (void *)(&dst[d_dex]);
    auto src_addr = (void *)(&src[s_dex]);
    // truncate hex address to 32 bits
    dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
    src_addr = (void *)((uint64_t)src_addr & 0xffffffff);
    // create memory trace for ramulator and write to file
    ofstream file;
    std::string filename =
        ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
    file.open(filename, std::ios_base::app);
    // file << (uint64_t) (src_addr++) << " R" << endl;
    // file << (uint64_t) (dst_addr++) << " W" << endl;
    uint64_t csrc = (uint64_t)(src_addr++);
    uint64_t cdst = (uint64_t)(dst_addr++);
    // modulo the address to 512 megabytes
    csrc = csrc % (512 * 1024 * 1024);
    cdst = cdst % (512 * 1024 * 1024);


    file << cpu_ins << " " << (uint64_t)(src_addr++) << " "
         << (uint64_t)(dst_addr++) << endl;

    // file << std::hex << (src_addr++) << " R" << endl;
    // file << std::hex << (dst_addr++) << " W" << endl;
    file.close();
  }

  template <typename T>
  void massign(T *dst, int d_dex) {
    auto dst_addr = (void *)(&dst[d_dex]);
    // truncate hex address to 32 bits
    dst_addr = (void *)((uint64_t)dst_addr & 0xffffffff);
    // create memory trace for ramulator and write to file
    ofstream file;
    std::string filename =
        ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
    file.open(filename, std::ios_base::app);
    file << std::hex << (dst_addr++) << " W" << endl;
    file.close();
  }

  void load_inject_dram_cycles() {
    if (use_sim) {
      // load latency from ramulator
      fstream file;
      std::string filename =
          ".data/secda_pim/layers/" + std::to_string(layer) + ".csv";
      file.open(filename, ios::in);

      // read header
      vector<string> row;
      std::string line, word, temp;
      int dram_cycles = 0;
      int mhz = 0;
      int cpu_cycles = 0;
      getline(file, line);
      istringstream s(line);
      char delim = ',';
      while (getline(s, word, delim)) {
        row.push_back(word);
      }
      dram_cycles = std::stoi(row[0]);
      mhz = std::stoi(row[1]);
      cpu_cycles = std::stoi(row[2]);

      // sc_start(dram_cycles * 1.85, SC_NS);
      sc_start(cpu_cycles * 1.54, SC_NS);

    }
  }

  acc_container(ACCNAME *_acc, int *_wb_0, int *_wb_1, int *_wb_2, int *_wb_3,
                std::vector<int> _wt_sum1, std::vector<int> _wt_sum2,
                std::vector<int> _wt_sum3, std::vector<int> _wt_sum4,
                std::vector<int> _crf, std::vector<int8_t> _crx) {
    acc = _acc;
    wb_0 = _wb_0;
    wb_1 = _wb_1;
    wb_2 = _wb_2;
    wb_3 = _wb_3;
    wt_sum1 = _wt_sum1;
    wt_sum2 = _wt_sum2;
    wt_sum3 = _wt_sum3;
    wt_sum4 = _wt_sum4;
    crf = _crf;
    crx = _crx;
  }
};

void preload_weights(int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        int8_t weights[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
      } else {
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
      }
    }
    wt_sum1.push_back(s0);
    wt_sum2.push_back(s1);
    wt_sum3.push_back(s2);
    wt_sum4.push_back(s3);
  }
}

#endif // ACC_CONTAINER