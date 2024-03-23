#ifndef SECDA_HW_UTILS_SC_H
#define SECDA_HW_UTILS_SC_H

#include <systemc.h>

template <typename T, unsigned int W>
struct sbram {
  T data[W];
  int size = W;
  int access_count = 0;
  int idx;
  sbram() {}
  T &operator[](int i) {
    idx = i;
    return data[i];
  }
  int &operator=(int val) { data[idx] = val; }
  void write(int i, T val) { data[i] = val; }
  T read(int i) { return data[i]; }
};

// sbam<sc_int<32>, 32> sram;

#endif // SECDA_HW_UTILS_SC_H