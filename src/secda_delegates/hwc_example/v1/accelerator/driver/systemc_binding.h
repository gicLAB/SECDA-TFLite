#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#ifdef SYSC

#include "../acc.sc.h"
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"

// This file is specfic to VADD SystemC definition
// This contains all the correct port/signal bindings to instantiate the VADD
// accelerator
struct sysC_sigs {
  int id;
  Clock_Reset_Define;
  sc_fifo<ADATA> dout1;
  sc_fifo<ADATA> din1;

  sysC_sigs(int id_) : dout1("dout1_fifo", 563840), din1("din1_fifo", 554800) {
    sc_clock clk_clock("ClkClock", 1, SC_NS);
    id = id_;
  }
};

void sysC_binder(ACCNAME *acc, sysC_sigs *scs, a_ctrl *ctrl, h_ctrl *hwc,
                 s_mdma *mdma) {

  Clock_Reset_Bind(acc, scs);
  Clock_Reset_Bind(ctrl->ctrl, scs);
  Clock_Reset_Bind(hwc->hwc_resetter, scs);

  CTRL_Bind_CtrlSignals(acc, ctrl);
  CTRL_Bind_RegSignals(testS);

  acc->dout1(scs->dout1);
  acc->din1(scs->din1);

  for (int i = 0; i < mdma->dma_count; i++) {
    mdma->dmas[i].dmad->clock(scs->clk_clock);
    mdma->dmas[i].dmad->reset(scs->sig_reset);
  }
  mdma->dmas[0].dmad->dout1(scs->dout1);
  mdma->dmas[0].dmad->din1(scs->din1);

  HWC_Bind_Reset;
  HWC_Bind_Signals(Recv);
  HWC_Bind_Signals(Compute);
  HWC_Bind_Signals(Send);
}
#endif // SYSC

#endif // SYSTEMC_BINDING
