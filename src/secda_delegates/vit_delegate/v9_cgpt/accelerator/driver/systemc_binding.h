#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

// #include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"


// This file is specfic to VitAcc SystemC definition
// This contains all the correct port/signal bindings to instantiate the VitAcc
// accelerator
struct sysC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  // sc_signal<int> sig_computeSS;
  sc_fifo<ADATA> dout1;
  sc_fifo<ADATA> dout2;
  sc_fifo<ADATA> dout3;
  sc_fifo<ADATA> dout4;
  
  sc_fifo<ADATA> din1;
  sc_fifo<ADATA> din2;
  sc_fifo<ADATA> din3;
  sc_fifo<ADATA> din4;

  sc_signal<int> sig_out; // out_sig


  int id; // Idk if out_sig has the right value here
  sysC_sigs(int _id)
      : dout1("dout1_fifo", 563840),
        dout2("dout2_fifo", 563840),
        dout3("dout3_fifo", 563840),
        dout4("dout4_fifo", 563840),
        din1("din1_fifo", 554800),
        din2("din2_fifo", 554800),
        din3("din3_fifo", 554800),
        din4("din4_fifo", 554800) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void sysC_binder(ACCNAME *acc, s_mdma *mdma, sysC_sigs *scs) {

  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->out_sig(scs->sig_out);

  for (int i = 0; i < mdma->dma_count; i++) {
    mdma->dmas[i].dmad->clock(scs->clk_fast);
    mdma->dmas[i].dmad->reset(scs->sig_reset);
  }
  mdma->dmas[0].dmad->dout1(scs->dout1);
  mdma->dmas[1].dmad->dout1(scs->dout2);
  mdma->dmas[2].dmad->dout1(scs->dout3);
  mdma->dmas[3].dmad->dout1(scs->dout4);
  mdma->dmas[0].dmad->din1(scs->din1);
  mdma->dmas[1].dmad->din1(scs->din2);
  mdma->dmas[2].dmad->din1(scs->din3);
  mdma->dmas[3].dmad->din1(scs->din4);

  acc->dout1(scs->dout1);
  acc->dout2(scs->dout2);
  acc->dout3(scs->dout3);
  acc->dout4(scs->dout4);
  acc->din1(scs->din1);
  acc->din2(scs->din2);
  acc->din3(scs->din3);
  acc->din4(scs->din4);

  // acc->computeSS(scs->sig_computeSS);
}

#endif // SYSTEMC_BINDING