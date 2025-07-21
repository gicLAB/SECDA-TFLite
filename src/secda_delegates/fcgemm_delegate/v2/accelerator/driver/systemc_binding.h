#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"

// This file is specfic to FC-GEMM SystemC definition
// This contains all the correct port/signal bindings to instantiate the FC-GEMM
// accelerator
struct sysC_sigs {
  int id;

  sc_clock clk_fast;
  sc_signal<bool> sig_reset;

  sc_signal<unsigned int> sig_insn_count;
  sc_signal<unsigned int> sig_insn_addr;
  sc_signal<unsigned int> sig_input_addr;
  sc_signal<unsigned int> sig_weight_addr;
  sc_signal<unsigned int> sig_bias_addr;
  sc_signal<unsigned int> sig_output_addr;

  sc_signal<int> sig_depth;
  sc_signal<int> sig_crf;
  sc_signal<int> sig_crx;
  sc_signal<int> sig_ra;

  CTRL_Define_Signals

  sysC_sigs(int _id) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id = _id;
  }
};

void sysC_binder(ACCNAME *acc, sysC_sigs *scs, a_ctrl *ctrl, mm_buf *insn_mem,
                 mm_buf *inp_mem, mm_buf *wgt_mem, mm_buf *bias_mem,
                 mm_buf2 *out_mem) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->insn_count(scs->sig_insn_count);
  acc->depth(scs->sig_depth);
  acc->crf(scs->sig_crf);
  acc->crx(scs->sig_crx);
  acc->ra(scs->sig_ra);
  acc->insn_addr(scs->sig_insn_addr);
  acc->input_addr(scs->sig_input_addr);
  acc->weight_addr(scs->sig_weight_addr);
  acc->bias_addr(scs->sig_bias_addr);
  acc->output_addr(scs->sig_output_addr);
  CTRL_Bind_Signals(acc, scs);
  acc->insn_port(insn_mem->buffer_chn);
  acc->input_port(inp_mem->buffer_chn);
  acc->weight_port(wgt_mem->buffer_chn);
  acc->bias_port(bias_mem->buffer_chn);
  acc->output_port(out_mem->buffer_chn);

  ctrl->ctrl->clock(scs->clk_fast);
  ctrl->ctrl->reset(scs->sig_reset);
  CTRL_Bind_Signals(ctrl->ctrl, scs);
}

#endif // SYSTEMC_BINDING
