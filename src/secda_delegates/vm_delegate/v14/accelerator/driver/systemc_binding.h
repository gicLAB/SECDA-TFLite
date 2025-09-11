#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

// This file is specfic to VM SystemC definition
// This contains all the correct port/signal bindings to instantiate the VM
// accelerator
struct sysC_sigs {
  int id;
  sc_clock clk_fast;
  sc_signal<bool> sig_reset;
  sc_signal<int> sig_inS;
  sc_signal<int> sig_outS;
  sc_signal<int> sig_schS;
  sc_signal<int> sig_p1S;
  sc_signal<int> sig_read_cycle_count;
  sc_signal<int> sig_process_cycle_count;

  sc_signal<int> sig_w1S;
  sc_signal<int> sig_w2S;
  sc_signal<int> sig_w3S;
  sc_signal<int> sig_w4S;
  sc_signal<int> sig_w5S;
  sc_signal<int> sig_w6S;
  sc_signal<int> sig_w7S;
  sc_signal<int> sig_w8S;
  sc_signal<int> sig_w9S;
  sc_signal<int> sig_w10S;
  sc_signal<int> sig_w11S;
  sc_signal<int> sig_w12S;
  sc_signal<int> sig_w13S;
  sc_signal<int> sig_w14S;
  sc_signal<int> sig_w15S;
  sc_signal<int> sig_w16S;

  sc_signal<int> sig_gemm_1_idle;
  sc_signal<int> sig_gemm_2_idle;
  sc_signal<int> sig_gemm_3_idle;
  sc_signal<int> sig_gemm_4_idle;
  sc_signal<int> sig_gemm_5_idle;
  sc_signal<int> sig_gemm_6_idle;
  sc_signal<int> sig_gemm_7_idle;
  sc_signal<int> sig_gemm_8_idle;
  sc_signal<int> sig_gemm_9_idle;
  sc_signal<int> sig_gemm_10_idle;
  sc_signal<int> sig_gemm_11_idle;
  sc_signal<int> sig_gemm_12_idle;
  sc_signal<int> sig_gemm_13_idle;
  sc_signal<int> sig_gemm_14_idle;
  sc_signal<int> sig_gemm_15_idle;
  sc_signal<int> sig_gemm_16_idle;

  sc_signal<int> sig_gemm_1_write;
  sc_signal<int> sig_gemm_2_write;
  sc_signal<int> sig_gemm_3_write;
  sc_signal<int> sig_gemm_4_write;
  sc_signal<int> sig_gemm_5_write;
  sc_signal<int> sig_gemm_6_write;
  sc_signal<int> sig_gemm_7_write;
  sc_signal<int> sig_gemm_8_write;
  sc_signal<int> sig_gemm_9_write;
  sc_signal<int> sig_gemm_10_write;
  sc_signal<int> sig_gemm_11_write;
  sc_signal<int> sig_gemm_12_write;
  sc_signal<int> sig_gemm_13_write;
  sc_signal<int> sig_gemm_14_write;
  sc_signal<int> sig_gemm_15_write;
  sc_signal<int> sig_gemm_16_write;

  sc_signal<int> sig_gemm_1;
  sc_signal<int> sig_gemm_2;
  sc_signal<int> sig_gemm_3;
  sc_signal<int> sig_gemm_4;
  sc_signal<int> sig_gemm_5;
  sc_signal<int> sig_gemm_6;
  sc_signal<int> sig_gemm_7;
  sc_signal<int> sig_gemm_8;
  sc_signal<int> sig_gemm_9;
  sc_signal<int> sig_gemm_10;
  sc_signal<int> sig_gemm_11;
  sc_signal<int> sig_gemm_12;
  sc_signal<int> sig_gemm_13;
  sc_signal<int> sig_gemm_14;
  sc_signal<int> sig_gemm_15;
  sc_signal<int> sig_gemm_16;

  sc_signal<int> sig_wstall_1;
  sc_signal<int> sig_wstall_2;
  sc_signal<int> sig_wstall_3;
  sc_signal<int> sig_wstall_4;
  sc_signal<int> sig_wstall_5;
  sc_signal<int> sig_wstall_6;
  sc_signal<int> sig_wstall_7;
  sc_signal<int> sig_wstall_8;
  sc_signal<int> sig_wstall_9;
  sc_signal<int> sig_wstall_10;
  sc_signal<int> sig_wstall_11;
  sc_signal<int> sig_wstall_12;
  sc_signal<int> sig_wstall_13;
  sc_signal<int> sig_wstall_14;
  sc_signal<int> sig_wstall_15;
  sc_signal<int> sig_wstall_16;

  sc_signal<int> sig_computeS0;
  sc_signal<int> sig_computeS1;
  sc_signal<int> sig_computeS2;
  sc_signal<int> sig_computeS3;
  sc_signal<int> sig_computeS4;
  sc_signal<int> sig_computeS5;
  sc_signal<int> sig_computeS6;
  sc_signal<int> sig_computeS7;
  sc_signal<int> sig_computeS8;
  sc_signal<int> sig_computeS9;
  sc_signal<int> sig_computeS10;
  sc_signal<int> sig_computeS11;
  sc_signal<int> sig_computeS12;
  sc_signal<int> sig_computeS13;
  sc_signal<int> sig_computeS14;
  sc_signal<int> sig_computeS15;

  sc_signal<int> sig_postS0;
  sc_signal<int> sig_postS1;
  sc_signal<int> sig_postS2;
  sc_signal<int> sig_postS3;
  sc_signal<int> sig_postS4;
  sc_signal<int> sig_postS5;
  sc_signal<int> sig_postS6;
  sc_signal<int> sig_postS7;
  sc_signal<int> sig_postS8;
  sc_signal<int> sig_postS9;
  sc_signal<int> sig_postS10;
  sc_signal<int> sig_postS11;
  sc_signal<int> sig_postS12;
  sc_signal<int> sig_postS13;
  sc_signal<int> sig_postS14;
  sc_signal<int> sig_postS15;

  sc_fifo<DATA> dout1;
  sc_fifo<DATA> dout2;
  sc_fifo<DATA> dout3;
  sc_fifo<DATA> dout4;

  sc_fifo<DATA> din1;
  sc_fifo<DATA> din2;
  sc_fifo<DATA> din3;
  sc_fifo<DATA> din4;

  sysC_sigs(int id_)
      : dout1("dout1_fifo", 563840), dout2("dout2_fifo", 563840),
        dout3("dout3_fifo", 563840), dout4("dout4_fifo", 563840),
        din1("din1_fifo", 554800), din2("din2_fifo", 554800),
        din3("din3_fifo", 554800), din4("din4_fifo", 554800),
        clk_fast("ClkFast", 5, SC_NS) {
    id = id_;
  }
};
;

void sysC_binder(ACCNAME *acc, multi_dma *mdma, sysC_sigs *scs) {
  acc->clock(scs->clk_fast);
  acc->reset(scs->sig_reset);
  acc->inS(scs->sig_inS);
  acc->outS(scs->sig_outS);

  acc->schS(scs->sig_schS);
  acc->p1S(scs->sig_p1S);

  acc->read_cycle_count(scs->sig_read_cycle_count);
  acc->process_cycle_count(scs->sig_process_cycle_count);

  acc->w1SS(scs->sig_w1S);
  acc->w2SS(scs->sig_w2S);
  acc->w3SS(scs->sig_w3S);
  acc->w4SS(scs->sig_w4S);
  acc->w5SS(scs->sig_w5S);
  acc->w6SS(scs->sig_w6S);
  acc->w7SS(scs->sig_w7S);
  acc->w8SS(scs->sig_w8S);
  acc->w9SS(scs->sig_w9S);
  acc->w10SS(scs->sig_w10S);
  acc->w11SS(scs->sig_w11S);
  acc->w12SS(scs->sig_w12S);
  acc->w13SS(scs->sig_w13S);
  acc->w14SS(scs->sig_w14S);
  acc->w15SS(scs->sig_w15S);
  acc->w16SS(scs->sig_w16S);

  acc->gemm_1_idle(scs->sig_gemm_1_idle);
  acc->gemm_2_idle(scs->sig_gemm_2_idle);
  acc->gemm_3_idle(scs->sig_gemm_3_idle);
  acc->gemm_4_idle(scs->sig_gemm_4_idle);
  acc->gemm_5_idle(scs->sig_gemm_5_idle);
  acc->gemm_6_idle(scs->sig_gemm_6_idle);
  acc->gemm_7_idle(scs->sig_gemm_7_idle);
  acc->gemm_8_idle(scs->sig_gemm_8_idle);
  acc->gemm_9_idle(scs->sig_gemm_9_idle);
  acc->gemm_10_idle(scs->sig_gemm_10_idle);
  acc->gemm_11_idle(scs->sig_gemm_11_idle);
  acc->gemm_12_idle(scs->sig_gemm_12_idle);
  acc->gemm_13_idle(scs->sig_gemm_13_idle);
  acc->gemm_14_idle(scs->sig_gemm_14_idle);
  acc->gemm_15_idle(scs->sig_gemm_15_idle);
  acc->gemm_16_idle(scs->sig_gemm_16_idle);

  acc->gemm_1_write(scs->sig_gemm_1_write);
  acc->gemm_2_write(scs->sig_gemm_2_write);
  acc->gemm_3_write(scs->sig_gemm_3_write);
  acc->gemm_4_write(scs->sig_gemm_4_write);
  acc->gemm_5_write(scs->sig_gemm_5_write);
  acc->gemm_6_write(scs->sig_gemm_6_write);
  acc->gemm_7_write(scs->sig_gemm_7_write);
  acc->gemm_8_write(scs->sig_gemm_8_write);
  acc->gemm_9_write(scs->sig_gemm_9_write);
  acc->gemm_10_write(scs->sig_gemm_10_write);
  acc->gemm_11_write(scs->sig_gemm_11_write);
  acc->gemm_12_write(scs->sig_gemm_12_write);
  acc->gemm_13_write(scs->sig_gemm_13_write);
  acc->gemm_14_write(scs->sig_gemm_14_write);
  acc->gemm_15_write(scs->sig_gemm_15_write);
  acc->gemm_16_write(scs->sig_gemm_16_write);

  acc->gemm_1(scs->sig_gemm_1);
  acc->gemm_2(scs->sig_gemm_2);
  acc->gemm_3(scs->sig_gemm_3);
  acc->gemm_4(scs->sig_gemm_4);
  acc->gemm_5(scs->sig_gemm_5);
  acc->gemm_6(scs->sig_gemm_6);
  acc->gemm_7(scs->sig_gemm_7);
  acc->gemm_8(scs->sig_gemm_8);
  acc->gemm_9(scs->sig_gemm_9);
  acc->gemm_10(scs->sig_gemm_10);
  acc->gemm_11(scs->sig_gemm_11);
  acc->gemm_12(scs->sig_gemm_12);
  acc->gemm_13(scs->sig_gemm_13);
  acc->gemm_14(scs->sig_gemm_14);
  acc->gemm_15(scs->sig_gemm_15);
  acc->gemm_16(scs->sig_gemm_16);

  acc->wstall_1(scs->sig_wstall_1);
  acc->wstall_2(scs->sig_wstall_2);
  acc->wstall_3(scs->sig_wstall_3);
  acc->wstall_4(scs->sig_wstall_4);
  acc->wstall_5(scs->sig_wstall_5);
  acc->wstall_6(scs->sig_wstall_6);
  acc->wstall_7(scs->sig_wstall_7);
  acc->wstall_8(scs->sig_wstall_8);
  acc->wstall_9(scs->sig_wstall_9);
  acc->wstall_10(scs->sig_wstall_10);
  acc->wstall_11(scs->sig_wstall_11);
  acc->wstall_12(scs->sig_wstall_12);
  acc->wstall_13(scs->sig_wstall_13);
  acc->wstall_14(scs->sig_wstall_14);
  acc->wstall_15(scs->sig_wstall_15);
  acc->wstall_16(scs->sig_wstall_16);

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

  acc->vars.vars_0.computeS(scs->sig_computeS0);
  acc->vars.vars_0.postS(scs->sig_postS0);
#if VMM_COUNT > 1
  acc->vars.vars_1.computeS(scs->sig_computeS1);
  acc->vars.vars_1.postS(scs->sig_postS1);

#if VMM_COUNT > 2
  acc->vars.vars_2.computeS(scs->sig_computeS2);
  acc->vars.vars_2.postS(scs->sig_postS2);

  acc->vars.vars_3.computeS(scs->sig_computeS3);
  acc->vars.vars_3.postS(scs->sig_postS3);

#if VMM_COUNT > 4
  acc->vars.vars_4.computeS(scs->sig_computeS4);
  acc->vars.vars_4.postS(scs->sig_postS4);

  acc->vars.vars_5.computeS(scs->sig_computeS5);
  acc->vars.vars_5.postS(scs->sig_postS5);

#if VMM_COUNT > 6
  acc->vars.vars_6.computeS(scs->sig_computeS6);
  acc->vars.vars_6.postS(scs->sig_postS6);

  acc->vars.vars_7.computeS(scs->sig_computeS7);
  acc->vars.vars_7.postS(scs->sig_postS7);
#if VMM_COUNT > 8
  acc->vars.vars_8.computeS(scs->sig_computeS8);
  acc->vars.vars_8.postS(scs->sig_postS8);

  acc->vars.vars_9.computeS(scs->sig_computeS9);
  acc->vars.vars_9.postS(scs->sig_postS9);

  acc->vars.vars_10.computeS(scs->sig_computeS10);
  acc->vars.vars_10.postS(scs->sig_postS10);

  acc->vars.vars_11.computeS(scs->sig_computeS11);
  acc->vars.vars_11.postS(scs->sig_postS11);

  acc->vars.vars_12.computeS(scs->sig_computeS12);
  acc->vars.vars_12.postS(scs->sig_postS12);

  acc->vars.vars_13.computeS(scs->sig_computeS13);
  acc->vars.vars_13.postS(scs->sig_postS13);

  acc->vars.vars_14.computeS(scs->sig_computeS14);
  acc->vars.vars_14.postS(scs->sig_postS14);

  acc->vars.vars_15.computeS(scs->sig_computeS15);
  acc->vars.vars_15.postS(scs->sig_postS15);

#endif
#endif
#endif
#endif
#endif
}

#endif // SYSTEMC_BINDING