#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/ramulator_connector/ramulator_connect.h"

// This file is specfic to Ramulator_addAcc SystemC definition
// This contains all the correct port/signal bindings to instantiate the
// Ramulator_addAcc accelerator
struct sysC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  sc_fifo<DATA> dout1;
  sc_fifo<DATA> din1;

  // sc_signal<bool> mem_start_read_sig;
  // sc_signal<bool> mem_read_done_sig;
  // sc_signal<unsigned int> mem_r_addr_sig;
  // sc_signal<unsigned int> mem_r_length_sig;

  // sc_signal<bool> mem_start_write_sig;
  // sc_signal<bool> mem_write_done_sig;
  // sc_signal<unsigned int> mem_w_addr_sig;
  // sc_signal<unsigned int> mem_w_length_sig;

  int id;
  sysC_sigs(int _id) : dout1("dout1_fifo", 563840), din1("din1_fifo", 554800) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void systemC_binder(ACCNAME *acc, stream_dma *sdma, sysC_sigs *scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->dout1(scs->dout1);
  acc->din1(scs->din1);

  sdma->dmad->clock(scs->clk_fast);
  sdma->dmad->reset(scs->sig_reset);
  sdma->dmad->rm.dout1(scs->dout1);
  sdma->dmad->rm.din1(scs->din1);

  // sdma->dmad->rm.mem_start_read(scs->mem_start_read_sig);
  // sdma->dmad->rm.mem_read_done(scs->mem_read_done_sig);
  // sdma->dmad->rm.mem_r_addr_p(scs->mem_r_addr_sig);
  // sdma->dmad->rm.mem_r_length_p(scs->mem_r_length_sig);

  // sdma->dmad->rm.mem_start_write(scs->mem_start_write_sig);
  // sdma->dmad->rm.mem_write_done(scs->mem_write_done_sig);
  // sdma->dmad->rm.mem_w_addr_p(scs->mem_w_addr_sig);
  // sdma->dmad->rm.mem_w_length_p(scs->mem_w_length_sig);

  // rconn->phy->clock(scs->clk_fast);
  // rconn->phy->reset(scs->sig_reset);
  // rconn->phy->mem_start_read(scs->mem_start_read_sig);
  // rconn->phy->mem_read_done(scs->mem_read_done_sig);
  // rconn->phy->mem_r_addr_p(scs->mem_r_addr_sig);
  // rconn->phy->mem_r_length_p(scs->mem_r_length_sig);

  // rconn->phy->mem_start_write(scs->mem_start_write_sig);
  // rconn->phy->mem_write_done(scs->mem_write_done_sig);
  // rconn->phy->mem_w_addr_p(scs->mem_w_addr_sig);
  // rconn->phy->mem_w_length_p(scs->mem_w_length_sig);
}

#endif // SYSTEMC_BINDING