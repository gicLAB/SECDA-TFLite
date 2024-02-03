#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

// This file is specfic to AddAcc SystemC definition
// This contains all the correct port/signal bindings to instantiate the AddAcc
// accelerator
struct sysC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  sc_signal<int> sig_computeSS;
  sc_fifo<DATA> dout1;
  sc_fifo<DATA> din1;

  int id;
  sysC_sigs(int _id) : dout1("dout1_fifo", 563840), din1("din1_fifo", 554800) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void sysC_binder(ACCNAME *acc, multi_dma *mdma, sysC_sigs *scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->dout1(scs->dout1);
  acc->din1(scs->din1);
  acc->computeSS(scs->sig_computeSS);

  for (int i = 0; i < mdma->dma_count; i++) {
    mdma->dmas[i].dmad->clock(scs->clk_fast);
    mdma->dmas[i].dmad->reset(scs->sig_reset);
  }
  mdma->dmas[0].dmad->dout1(scs->dout1);
  mdma->dmas[0].dmad->din1(scs->din1);
}

#endif // SYSTEMC_BINDING