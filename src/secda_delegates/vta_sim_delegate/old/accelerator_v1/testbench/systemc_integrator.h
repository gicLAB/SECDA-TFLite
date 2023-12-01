
#ifndef SYSTEMC_INTEGRATION
#define SYSTEMC_INTEGRATION

#include <systemc.h>
#include "../hls_bus_if.h"
// #include "gen_instructions.h"
#include "vta_driver.h"



void sysC_init() {
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions(SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions(SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);
}

struct systemC_sigs {
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;

  hls_bus_chn<unsigned long long> insns_mem;
  hls_bus_chn<unsigned int> uops_mem;
  hls_bus_chn<unsigned long long> data_mem;

  sc_signal<unsigned int> sig_vtaS;
  sc_signal<unsigned int> sig_insn_count;
  sc_signal<unsigned int> sig_ins_addr;
  sc_signal<unsigned int> sig_uops_addr;
  sc_signal<unsigned int> sig_input_addr;
  sc_signal<unsigned int> sig_weight_addr;
  sc_signal<unsigned int> sig_bias_addr;
  sc_signal<unsigned int> sig_output_addr;

  sc_signal<unsigned int> sig_crf_addr;
  sc_signal<unsigned int> sig_crx_addr;
  sc_signal<unsigned int> sig_ra_sig;

  int id;

  systemC_sigs(int _id)
      : insns_mem("insns", 0, 40960),
        uops_mem("uops", 0, 8192),
        data_mem("data", 0, 409600) {
    sc_clock clk_fast("ClkFast", 1, SC_NS);
    id =_id;
  }
};

void systemC_binder(ACCNAME* acc, VTA_Driver* vdriver, int _insns_mem_size,
                    int _uops_mem_size, int _data_mem_size, systemC_sigs* scs) {

  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->vtaS(scs->sig_vtaS);
  acc->insn_count(scs->sig_insn_count);
  acc->ins_addr(scs->sig_ins_addr);
  acc->uops_addr(scs->sig_uops_addr);
  acc->input_addr(scs->sig_input_addr);
  acc->weight_addr(scs->sig_weight_addr);
  acc->bias_addr(scs->sig_bias_addr);
  acc->output_addr(scs->sig_output_addr);
  acc->crf_addr(scs->sig_crf_addr);
  acc->crx_addr(scs->sig_crx_addr);

  acc->ra_sig(scs->sig_ra_sig);

  acc->insns(scs->insns_mem);
  acc->uops(scs->uops_mem);
  acc->data(scs->data_mem);

  vdriver->clock(scs->clk_fast);
  vdriver->reset(scs->sig_reset);
  vdriver->vtaS(scs->sig_vtaS);
  vdriver->insn_count(scs->sig_insn_count);
  vdriver->ins_addr(scs->sig_ins_addr);
  vdriver->uops_addr(scs->sig_uops_addr);
  vdriver->input_addr(scs->sig_input_addr);
  vdriver->weight_addr(scs->sig_weight_addr);
  vdriver->bias_addr(scs->sig_bias_addr);
  vdriver->output_addr(scs->sig_output_addr);
  vdriver->insns(scs->insns_mem);
  vdriver->uops(scs->uops_mem);
  vdriver->data(scs->data_mem);

  // int rows = 16;
  // int cols = 16;
  // int depth = 64;
  // gemm_test(sig_insn_count, sig_ins_addr, sig_uops_addr, sig_input_addr,
  //           sig_weight_addr, sig_bias_addr, sig_output_addr, insns_mem,
  //           uops_mem, data_mem, rows, depth, cols, false);

  // int batch = 16;
  // int in_channels = 16;
  // int out_channels = 32;
  // int block = 16;

  // int batch = 1024;
  // int in_channels = 32;
  // int out_channels = 32;
  // int block = 32;

  // blocked_gemm_test(scs->sig_insn_count, scs->sig_ins_addr, scs->sig_uops_addr,
  //                   scs->sig_input_addr, scs->sig_weight_addr,
  //                   scs->sig_bias_addr, scs->sig_output_addr, scs->insns_mem,
  //                   scs->uops_mem, scs->data_mem, batch, in_channels,
  //                   out_channels, block, false, 1);

  // batch = 16;
  // in_channels = 32;
  // out_channels = 16;
  // block = 16;

  // blocked_gemm_test(scs->sig_insn_count, scs->sig_ins_addr, scs->sig_uops_addr,
  //                   scs->sig_input_addr, scs->sig_weight_addr,
  //                   scs->sig_bias_addr, scs->sig_output_addr, scs->insns_mem,
  //                   scs->uops_mem, scs->data_mem, batch, in_channels,
  //                   out_channels, block, false, 1);

      // blocked_gemm_test_tflite(
      //     scs->sig_insn_count, scs->sig_ins_addr, scs->sig_uops_addr,
      //     scs->sig_input_addr, scs->sig_weight_addr, scs->sig_bias_addr,
      //     scs->sig_output_addr, scs->insns_mem, scs->uops_mem, scs->data_mem,
      //     pN, pK, pM, 16, false, 1, padded_input, padded_weights,
      //     padded_bias, padded_output);
}

#endif  // SYSTEMC_INTEGRATION
