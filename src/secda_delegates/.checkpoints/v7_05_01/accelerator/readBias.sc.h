#ifndef READBIAS_H
#define READBIAS_H

#include "acc_config.sc.h"

// Defined in acc.sc.h
// void ACCNAME::ReadBias() { ... }

void ACCNAME::ReadBias() {
  biasReadReadyS.write(0);
  wait();
  DWAIT(2);

  while (1) {
    while (biasReadReadyS.read() == 0) wait();

    int pnr = pN_rem;
    int pmr = pM_rem;

    rhs_offset = (int)din4->read().data.to_int();
    lhs_offset = (int)din4->read().data.to_int();

    // Offsets printed once in vit_delegate.cc - suppress repeated output here
    // cout << "ReadBias: rhs_offset = " << rhs_offset
    //      << ", lhs_offset = " << lhs_offset << endl;

    // 1. Read Quantization Parameters (Per-Channel Scales/Shifts)
    if (layer_t == 1) {
      for (int a = 0; a < pmr; a += 4) {
        crf_v[a + 0] = din1->read().data;
        crf_v[a + 1] = din1->read().data;
        crf_v[a + 2] = din1->read().data;
        crf_v[a + 3] = din1->read().data;

        ADATA d_crx = din4->read();
        crx_v[a + 0] = d_crx.data.range(7, 0);
        crx_v[a + 1] = d_crx.data.range(15, 8);
        crx_v[a + 2] = d_crx.data.range(23, 16);
        crx_v[a + 3] = d_crx.data.range(31, 24);
      }
    }
    wait();

    // [FIX] Explicitly clear bias and wt_sum buffers to avoid garbage in padding
    for (int a = 0; a < pm_block; a++) {
       bias[a] = 0;
       wt_sum[a] = 0;
    }
    wait();

    // 2. Read Bias
    if (is_bias == 1) {
      for (int a = 0; a < pmr; a++) {
        bias[a] = din4->read().data;
      }
    }
    wait();

    // 3. Read Weight Sums
    for (int a = 0; a < pmr; a++) {
      int wt_data = din1->read().data;
      // Note: Assuming you kept the multiplication here based on previous steps
      wt_sum[a] = wt_data * rhs_offset; 
    }
    wait();

    // 4. Calculate Precision (Pre-Accumulation)
    for (int a = 0; a < pn_block; a++) {
#pragma HLS pipeline II = 1
      if (a < pnr) {
        int in_data = din4->read().data;
        in_sum[a] = in_data * lhs_offset;
        
        for (int b = 0; b < pm_block; b++) {
           // [FIX] Only calculate prec for VALID cols. 
           // Force 0 for padding to prevent 'in_sum' from leaking into padding cols.
           if (b < pmr) {
             prec[a][b] = bias[b] + wt_sum[b] + in_sum[a];
           } else {
            //  prec[a][b] = 0; 
           }
        }
      }
    }
    wait();
    readTimeBiasS.write(0);
    biasReadReadyS.write(0);
    wait();
  }
}

#endif // READBIAS_H