
#ifndef PHY_H
#define PHY_H

#include "../secda_integrator/sysc_types.h"
#include <iomanip>
#include <iostream>

SC_MODULE(PHY) {

  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<bool> mem_start_read;
  sc_in<unsigned int> mem_r_addr_p;
  sc_in<unsigned int> mem_r_length_p;
  sc_out<bool> mem_read_done;

  sc_in<bool> mem_start_write;
  sc_in<unsigned int> mem_w_addr_p;
  sc_in<unsigned int> mem_w_length_p;
  sc_out<bool> mem_write_done;

  sc_int<32> temp_in[10000];
  int layer = 0;

  int read_count = 0;
  int write_count = 0;

  void read() {
    DWAIT();
    while (1) {
      while (!mem_start_read.read()) DWAIT();
      unsigned int addr = mem_r_addr_p.read();
      unsigned int length = mem_r_length_p.read();

      // generate trace for ramulator
      ofstream file;
      std::string filename =
          ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
      file.open(filename, std::ios_base::app);
      int index = 0;
      for (int i = 0; i < length; i += 4) {
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 0) << " R" << endl;
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 1) << " R" << endl;
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 2) << " R" << endl;
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 3) << " R" << endl;
      }
      file.close();
      mem_read_done.write(1);
      while (mem_start_read.read()) DWAIT();
      mem_read_done.write(0);
      DWAIT();
    }
  };

  void write() {
    DWAIT();
    while (1) {
      while (!mem_start_write.read()) DWAIT();
      unsigned int addr = mem_w_addr_p.read();
      unsigned int length = mem_w_length_p.read();

      // generate trace for ramulator
      ofstream file;
      std::string filename =
          ".data/secda_pim/traces/" + std::to_string(layer) + ".trace";
      file.open(filename, std::ios_base::app);
      for (int i = 0; i < length; i += 4) {
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 0) << " W" << endl;
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 1) << " W" << endl;
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 2) << " W" << endl;
        file << "0x" << std::setfill('0') << std::setw(8) << std::hex
             << (addr + i + 3) << " W" << endl;
      }
      file.close();

      mem_write_done.write(1);
      while (mem_start_write.read()) DWAIT();
      mem_write_done.write(0);
      DWAIT();
    }
  };

  SC_HAS_PROCESS(PHY);

  PHY(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(read, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(write, clock.pos());
    reset_signal_is(reset, true);
  }
};

#endif