
#ifndef MM_CONTROLLER_H
#define MM_CONTROLLER_H

#include "../secda_integrator/sysc_types.h"
#include <iomanip>
#include <iostream>

SC_MODULE(MM_CONTROLLER) {

  sc_in<bool> clock;
  sc_in<bool> reset;

  sc_in<bool> r_en;
  sc_in<int> r_addr;
  sc_out<bool> rdone;

  sc_in<bool> w_en;
  sc_in<int> w_addr;
  sc_out<bool> wdone;

  unsigned int r_address[10000];
  unsigned int w_address[10000];
  unsigned int r_len;
  unsigned int w_len;


  // void read() {
  //   DWAIT();
  //   while (1) {

  //     while (!r_en.read()) DWAIT();
  //     int addr = r_addr.read();
  //     // generate trace for ramulator
  //     ofstream file;
  //     std::string filename = ".data/secda_pim/data/read.trace";
  //     file.open(filename, std::ios_base::app);
  //     file << "0x" << std::setfill('0') << std::setw(8) << std::hex << (addr)
  //          << " R" << endl;
  //     file.close();
  //     rdone.write(1);
  //     while (r_en.read()) DWAIT();
  //     rdone.write(0);
  //     DWAIT();
  //   }
  // };

  // void write() {
  //   DWAIT();
  //   while (1) {
  //     while (!w_en.read()) DWAIT();
  //     int addr = w_addr.read();
  //     // generate trace for ramulator
  //     ofstream file;
  //     std::string filename = ".data/secda_pim/data/write.trace";
  //     file.open(filename, std::ios_base::app);
  //     file << "0x" << std::setfill('0') << std::setw(8) << std::hex << (addr)
  //          << " W" << endl;
  //     file.close();
  //     wdone.write(1);
  //     while (w_en.read()) DWAIT();
  //     wdone.write(0);

  //     DWAIT();
  //   }
  // };

  SC_HAS_PROCESS(MM_CONTROLLER);

  MM_CONTROLLER(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(read, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(write, clock.pos());
    reset_signal_is(reset, true);
  }
};

#endif