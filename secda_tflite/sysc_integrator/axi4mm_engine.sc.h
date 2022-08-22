#ifndef AXI4MM_ENGINE_H
#define AXI4MM_ENGINE_H

#include <systemc.h>
#include "sysc_types.h"
#include "../ap_sysc/AXI4_if.h"

template <typename dtype>
SC_MODULE(AXI4MM_ENGINE) {
  sc_in<bool> clock;
  sc_in<bool> reset;


  AXI4M_bus_port<dtype> port;
  bool send;
  bool recv;

  // TODO: added logic to only write if burst_write
  void Write() {
    while (1) {
      port.burst_write(write_offset, write_len, (dtype*)&DMA_input_buffer[0]);
      wait();
    }
  };

  // TODO: added logic to only write if burst_read
  void Read() {
    while (1) {
      port.burst_read(read_offset, read_len, (dtype*)&DMA_output_buffer[0]);
      wait();

    }
  };

  SC_HAS_PROCESS(AXI4MM_ENGINE);

  AXI4MM_ENGINE(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Write, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Read, clock.pos());
    reset_signal_is(reset, true);
  }

  bool burst_write;
  int write_len;
  int write_offset;
  int *DMA_input_buffer;

  bool burst_read;
  int read_len;
  int read_offset;
  int *DMA_output_buffer;
};

#endif