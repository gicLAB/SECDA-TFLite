#ifndef SYSC_TYPES_H
#define SYSC_TYPES_H

#include <iomanip>
#include <iostream>
#include <systemc.h>
#include "secda_hw_utils.sc.h"

#ifndef DWAIT(x)
#ifndef __SYNTHESIS__
#define DWAIT(x) wait(x)
#else
#define DWAIT(x)
#endif
#endif

#define INITSIGPORT(X, SID) X((std::string(#X) + std::to_string(SID)).c_str())

// Hardware struct to contain output signal and port
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
  void pack(sc_int<8> a1, sc_int<8> a2, sc_int<8> a3, sc_int<8> a4) {
    data.range(7, 0) = a1;
    data.range(15, 8) = a2;
    data.range(23, 16) = a3;
    data.range(31, 24) = a4;
  }
} DATA;

typedef struct _SDATA {
  sc_int<32> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _SDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} SDATA;

template <int W>
struct _FDATA {
  sc_uint<W> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _FDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
};

struct rm_data2 {
  sc_fifo_in<DATA> dout1;
  sc_fifo_out<DATA> din1;

  // int base_r_addr = 0;
  // int mem_r_addr = 0;
  // sc_out<bool> mem_start_read;
  // sc_in<bool> mem_read_done;
  // sc_out<unsigned int> mem_r_addr_p;
  // sc_out<unsigned int> mem_r_length_p;

  // int base_w_addr = 0;
  // int mem_w_addr = 0;
  // sc_out<bool> mem_start_write;
  // sc_in<bool> mem_write_done;
  // sc_out<unsigned int> mem_w_addr_p;
  // sc_out<unsigned int> mem_w_length_p;

  bool use_sim = false;
  int layer = 0;

  // this writes to acc and reads from main memory

  void generate_reads(int addr, int length) {
    // generate trace for ramulator
    ofstream file;
    std::string filename =
        ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
    file.open(filename, std::ios_base::app);
    int index = 0;
    for (int i = 0; i < length; i += 4) {
      file << "0x" << std::setfill('0') << std::setw(8) << std::hex
           << (addr + i + 0) << " R" << endl;
      // file << "0x" << std::setfill('0') << std::setw(8) << std::hex
      //      << (addr + i + 1) << " R" << endl;
      // file << "0x" << std::setfill('0') << std::setw(8) << std::hex
      //      << (addr + i + 2) << " R" << endl;
      // file << "0x" << std::setfill('0') << std::setw(8) << std::hex
      //      << (addr + i + 3) << " R" << endl;
    }
    file.close();
  }

  void generate_writes(int addr, int length) {
    // generate trace for ramulator
    ofstream file;
    std::string filename =
        ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
    file.open(filename, std::ios_base::app);
    int index = 0;
    for (int i = 0; i < length; i += 4) {
      file << "0x" << std::setfill('0') << std::setw(8) << std::hex
           << (addr + i + 0) << " W" << endl;
      // file << "0x" << std::setfill('0') << std::setw(8) << std::hex
      //      << (addr + i + 1) << " W" << endl;
      // file << "0x" << std::setfill('0') << std::setw(8) << std::hex
      //      << (addr + i + 2) << " W" << endl;
      // file << "0x" << std::setfill('0') << std::setw(8) << std::hex
      //      << (addr + i + 3) << " W" << endl;
    }
    file.close();
  }

  void write(DATA d, int r_addr) {
    if (!use_sim) {
      // generate_reads(r_addr, 4);
      // mem_start_read.write(true);
      // mem_r_addr_p.write(r_addr);
      // mem_r_length_p.write(4);
      // DWAIT(1);
      // while (!mem_read_done.read()) DWAIT(1);
      // mem_start_read.write(false);
      // DWAIT(1);
    }
    din1.write(d);
  }

  // this reads from acc and writes to the main memory
  DATA read(int w_addr) {
    if (!use_sim) {
      // generate_writes(w_addr, 4);
      // mem_start_write.write(true);
      // mem_w_addr_p.write(w_addr);
      // mem_w_length_p.write(4);
      // DWAIT(1);
      // while (!mem_write_done.read()) DWAIT(1);
      // mem_start_write.write(false);
      // DWAIT(1);
    }
    return dout1.read();
  }
};

template <int W>
using FDATA = _FDATA<W>;

#endif