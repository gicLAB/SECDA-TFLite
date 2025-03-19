#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"

// This file is specfic to VM SystemC definition
// This contains all the correct port/signal bindings to instantiate the VM
// accelerator
struct sysC_sigs {
  int id;
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;

  sc_fifo<ADATA> dout1;
  sc_fifo<ADATA> dout2;
  sc_fifo<ADATA> dout3;
  sc_fifo<ADATA> dout4;

  sc_fifo<ADATA> din1;
  sc_fifo<ADATA> din2;
  sc_fifo<ADATA> din3;
  sc_fifo<ADATA> din4;

  sc_signal<bool> on;
  sc_signal<int> inS;
  sc_signal<int> data_loadS;
  sc_signal<int> scheduleS;
  sc_signal<int> outS;
  sc_signal<int> tempS;
  sc_signal<int> pdS;

  sc_signal<int> computeS0;
  sc_signal<int> sendS0;
  sc_signal<int> computeS1;
  sc_signal<int> sendS1;
  sc_signal<int> computeS2;
  sc_signal<int> sendS2;
  sc_signal<int> computeS3;
  sc_signal<int> sendS3;
  sc_signal<int> computeS4;
  sc_signal<int> sendS4;
  sc_signal<int> computeS5;
  sc_signal<int> sendS5;
  sc_signal<int> computeS6;
  sc_signal<int> sendS6;
  sc_signal<int> computeS7;
  sc_signal<int> sendS7;
  

  sysC_sigs(int id_)
      : dout1("dout1_fifo", 563840), dout2("dout2_fifo", 563840),
        dout3("dout3_fifo", 563840), dout4("dout4_fifo", 563840),
        din1("din1_fifo", 554800), din2("din2_fifo", 554800),
        din3("din3_fifo", 554800), din4("din4_fifo", 554800) {
    id = id_;
    sc_clock clk_fast("ClkFast", 1, SC_NS);
  }
};

void sysC_binder(ACCNAME *acc, s_mdma *mdma, sysC_sigs *scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);

  for (int i = 0; i < mdma->dma_count; i++) {
    mdma->dmas[i].dmad->clock(scs->clk_fast);
    mdma->dmas[i].dmad->reset(scs->sig_reset);
  }
  mdma->dmas[0].dmad->dout1(scs->dout1);
  // mdma->dmas[1].dmad->dout1(scs->dout2);
  // mdma->dmas[2].dmad->dout1(scs->dout3);
  // mdma->dmas[3].dmad->dout1(scs->dout4);
  mdma->dmas[0].dmad->din1(scs->din1);
  // mdma->dmas[1].dmad->din1(scs->din2);
  // mdma->dmas[2].dmad->din1(scs->din3);
  // mdma->dmas[3].dmad->din1(scs->din4);

  acc->dout1(scs->dout1);
  acc->dout2(scs->dout2);
  acc->dout3(scs->dout3);
  acc->dout4(scs->dout4);
  acc->din1(scs->din1);
  acc->din2(scs->din2);
  acc->din3(scs->din3);
  acc->din4(scs->din4);
  acc->on(scs->on);

  acc->inS(scs->inS);
  acc->data_loadS(scs->data_loadS);
  acc->scheduleS(scs->scheduleS);
  acc->outS(scs->outS);
  acc->tempS(scs->tempS);
  acc->pdS(scs->pdS);

  acc->vars.vars_0.computeS(scs->computeS0);
  acc->vars.vars_0.sendS(scs->sendS0);
  acc->vars.vars_1.computeS(scs->computeS1);
  acc->vars.vars_1.sendS(scs->sendS1);
  acc->vars.vars_2.computeS(scs->computeS2);
  acc->vars.vars_2.sendS(scs->sendS2);
  acc->vars.vars_3.computeS(scs->computeS3);
  acc->vars.vars_3.sendS(scs->sendS3);
  acc->vars.vars_4.computeS(scs->computeS4);
  acc->vars.vars_4.sendS(scs->sendS4);
  acc->vars.vars_5.computeS(scs->computeS5);
  acc->vars.vars_5.sendS(scs->sendS5);
  acc->vars.vars_6.computeS(scs->computeS6);
  acc->vars.vars_6.sendS(scs->sendS6);
  acc->vars.vars_7.computeS(scs->computeS7);
  acc->vars.vars_7.sendS(scs->sendS7);
}

#endif // SYSTEMC_BINDING