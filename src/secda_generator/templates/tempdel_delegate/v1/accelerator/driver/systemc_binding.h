#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"

// This file is specfic to TempdelAcc SystemC definition
// This contains all the correct port/signal bindings to instantiate the TempdelAcc
// accelerator
struct sysC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  sc_signal<int> sig_computeSS;
  sc_fifo<ADATA> dout1;
  sc_fifo<ADATA> din1;

  int id;
  sysC_sigs(int _id) : dout1("dout1_fifo", 563840), din1("din1_fifo", 554800) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void sysC_binder(ACCNAME *acc, s_mdma *mdma, sysC_sigs *scs) {
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