#include "acc_config.sc.h"

void ACCNAME::ReadBias() {
  wait();

  while (1) {
    while (biasReadReadyS.read() == 0) wait();

    readTimeBiasS.write(1);

    sc_int<8> current_mode = mode.read();
    rhs_offset = din4->read().data;
    lhs_offset = din4->read().data;

    sc_int<8> layer_type = layer_t;

    DWAIT(2);

    if (layer_type == 1) {
      if (current_mode == MODE_DENSE) {
        for (int core = 0; core < NUM_CORES; core++) {
          for (int a = 0; a < pm_block; a += 4) {
            crf_v[core][a + 0] = din1->read().data;
            crf_v[core][a + 1] = din1->read().data;
            crf_v[core][a + 2] = din1->read().data;
            crf_v[core][a + 3] = din1->read().data;

            ADATA d_crx = din4->read();
            crx_v[core][a + 0] = d_crx.data.range(7, 0);
            crx_v[core][a + 1] = d_crx.data.range(15, 8);
            crx_v[core][a + 2] = d_crx.data.range(23, 16);
            crx_v[core][a + 3] = d_crx.data.range(31, 24);
            DWAIT(4);
          }
        }
      } else { 
        for (int a = 0; a < pm_block; a += 4) {
          int v0 = din1->read().data;
          int v1 = din1->read().data;
          int v2 = din1->read().data;
          int v3 = din1->read().data;
          ADATA d_crx = din4->read();
          DWAIT(4);
          for (int core = 0; core < NUM_CORES; core++) {
            crf_v[core][a + 0] = v0;
            crf_v[core][a + 1] = v1;
            crf_v[core][a + 2] = v2;
            crf_v[core][a + 3] = v3;
            
            crx_v[core][a + 0] = d_crx.data.range(7, 0);
            crx_v[core][a + 1] = d_crx.data.range(15, 8);
            crx_v[core][a + 2] = d_crx.data.range(23, 16);
            crx_v[core][a + 3] = d_crx.data.range(31, 24);
            DWAIT(1);
          }
        }
      }
    }
    wait();

    if (is_bias == 1) {
      if (current_mode == MODE_DENSE) {
        for (int core = 0; core < NUM_CORES; core++) {
          for (int a = 0; a < pm_block; a++) bias[core][a] = din4->read().data;
        }
      } else {
        for (int a = 0; a < pm_block; a++) {
          int val = din4->read().data;
          for (int core = 0; core < NUM_CORES; core++) bias[core][a] = val;
        }
      }
    } else {
      for (int core = 0; core < NUM_CORES; core++)
        for (int a = 0; a < pm_block; a++) bias[core][a] = 0;
    }
    DWAIT(4);
    wait();

    if (current_mode == MODE_DENSE) {
      for (int core = 0; core < NUM_CORES; core++) {
        for (int a = 0; a < pm_block; a++)
          wt_sum[core][a] = din1->read().data * rhs_offset;
      }
    } else {
      for (int a = 0; a < pm_block; a++) {
        int val = din1->read().data * rhs_offset;
        for (int core = 0; core < NUM_CORES; core++) wt_sum[core][a] = val;
      }
    }
    DWAIT(6);
    wait();

    if (current_mode == MODE_DENSE) {
      for (int a = 0; a < pn_block; a++) {
        int i_sum = din4->read().data * lhs_offset;
        for (int core = 0; core < NUM_CORES; core++) {
          in_sum[core][a] = i_sum;
        }
      }
    } else {
      for (int core = 0; core < NUM_CORES; core++) {
        for (int a = 0; a < pn_block; a++) {
          int i_sum = din4->read().data * lhs_offset;
          in_sum[core][a] = i_sum;
        }
      }
    }
    DWAIT(12);

    wait();
    biasReadReadyS.write(0);
    readTimeBiasS.write(0);
    wait();
  }
}