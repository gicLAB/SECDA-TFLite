#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include "../acc.sc.h"
#include "systemc_binding.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#include <sstream>
#include <vector>

#include <fstream>
#include <iostream>
#include <string>

struct acc_container {
  // Hardware
  struct sysC_sigs *scs;
  Profile *profile;
  ACCNAME *acc;
  struct stream_dma *sdma;

  // Data
  int length;
  const int8_t *input_A;
  const int8_t *input_B;
  int8_t *output_C;

  // PPU
  int lshift;
  int in1_off;
  int in1_sv;
  int in1_mul;
  int in2_off;
  int in2_sv;
  int in2_mul;
  int out1_off;
  int out1_sv;
  int out1_mul;
  int qa_max;
  int qa_min;

  // Debugging
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
      file << std::hex << (src_addr++) << " R" << endl;
      file << std::hex << (src_addr++) << " R" << endl;
      file << std::hex << (src_addr++) << " R" << endl;
      file << std::hex << (dst_addr++) << " W" << endl;
      file << std::hex << (dst_addr++) << " W" << endl;
      file << std::hex << (dst_addr++) << " W" << endl;
      file << std::hex << (dst_addr++) << " W" << endl;
      file.close();
    }
    dst[d_dex] = value;
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
      getline(file, line);
      istringstream s(line);
      char delim = ',';
      while (getline(s, word, delim)) {
        row.push_back(word);
      }
      dram_cycles = std::stoi(row[0]);
      mhz = std::stoi(row[1]);
      sc_start(dram_cycles, SC_NS);
    }
  }
};

#endif // ACC_CONTAINER