#ifndef DRIVER_NAME
#define DRIVER_NAME

// CORRECT

#include "acc_container.h"

namespace vit_sim {

void block_add(acc_container &drv) {

  // prf_start(2); // fpga_total
  // cout << "int is " << sizeof(int) << endl; // 4 bits

  int param_len = 0;
  // dib = dibfer
  int *param_buf =
      drv.mdma->dmas[0].dma_get_inbuffer(); // returns int pointer (32 bit)

  int input_len = 0;
  int *input_buf =
      drv.mdma->dmas[1].dma_get_inbuffer();

  int weight_len = 0;
  int *weight_buf =
      drv.mdma->dmas[2].dma_get_inbuffer(); 

  int bias_len = 0;
  int *bias_buf =
      drv.mdma->dmas[3].dma_get_inbuffer(); 

  param_buf[param_len++] = drv.pN;
  param_buf[param_len++] = drv.pM;
  param_buf[param_len++] = drv.pK;

  param_buf[param_len++] = drv.crx;
  param_buf[param_len++] = drv.crf;
  param_buf[param_len++] = drv.ra;

  int NO_ROWS = pnF;
  int NO_COLS = pmF;

  param_buf[param_len++] = NO_ROWS;
  param_buf[param_len++] = NO_COLS;

  // cout << "(V3) Layer info: " << "PN: " << drv.pN << " PM: " << drv.pM
  //      << " PK: " << drv.pK << " CRX: " << drv.crx << " CRF: " << drv.crf
  //      << " RA: " << drv.ra << " NO_ROWS: " << NO_ROWS
  //      << " NO_COLS: " << NO_COLS << endl;

  // cout << "ViT_ACC (V3)" << endl;

  drv.mdma->dmas[0].dma_start_send(param_len);
  // drv.mdma->dmas[0].dma_wait_send();

  param_len = 0;

  for (int n = 0; n < drv.pN; n += NO_ROWS) {
    int n_remaining = min(NO_COLS, drv.pN - n);
    input_buf[input_len++] = n_remaining;
    memcpy(input_buf + input_len, drv.padded_input + n * drv.pK, 
       drv.pK * n_remaining * sizeof(int8_t));
    input_len += (drv.pK * n_remaining) / 4;

    // TODO: Is the issue that I send and wait ?
    drv.mdma->dmas[1].dma_start_send(input_len);
    // drv.mdma->dmas[1].dma_wait_send();
    input_len = 0;

    for (int m = 0; m < drv.pM; m += NO_COLS) {
      int m_remaining = min(NO_COLS, drv.pM - m);
      weight_buf[weight_len++] = m_remaining;

      memcpy(weight_buf + weight_len, drv.padded_weights + m * drv.pK,
             drv.pK * m_remaining * sizeof(int8_t));
      weight_len += (drv.pK * m_remaining) / 4;

      drv.mdma->dmas[2].dma_start_send(weight_len);
      // drv.mdma->dmas[2].dma_wait_send();
      weight_len = 0;


 
      for (int a = n; a < n + n_remaining; a++) {
        for (int b = m; b < m + m_remaining; b++) {
       //    cout << "(v2)bias: " << drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
       //                       (drv.in_sum[a] * (drv.lhs_offset)) << endl;
              bias_buf[bias_len++] = drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
                             (drv.in_sum[a] * (drv.lhs_offset));

        }
      }

      prf_start(0); // fpga_total
      drv.mdma->dmas[3].dma_start_send(bias_len);
      bias_len = 0;
      drv.mdma->multi_dma_wait_send();
      drv.mdma->dmas[0].dma_start_recv(n_remaining * m_remaining /
                                       4);
      drv.mdma->dmas[0].dma_wait_recv();
      prf_end(0, drv.a_t->fpga_total); // fpga_total

      param_len = 0;
      input_len = 0;
      weight_len = 0;
      bias_len = 0;

      int8_t *output_val =
          reinterpret_cast<int8_t *>(drv.mdma->dmas[0].dma_get_outbuffer());
      for (int i = 0; i < n_remaining; i++) {
        int base = n * drv.pM + m;
        int offset = i * drv.pM;
        int output_offset = i * m_remaining;


        memcpy(drv.padded_output + (base + offset), output_val + output_offset,
               m_remaining * sizeof(int8_t));
      }
    }
  }
}

void Entry(acc_container &drv) {
#ifdef DELEGATE_VERBOSE
  cout << "FC ACC - Layer: " << drv.layer << endl;
  cout << "===============";
  cout << "Pre-ACC Info" << endl;
  // cout << endl;
  cout << "padded_K: " << drv.pK << "K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << "M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << "N: " << drv.N << endl;
  cout << "===========================";
#endif

  block_add(drv);
}
} // namespace vit_sim

#endif // DRIVER_NAME