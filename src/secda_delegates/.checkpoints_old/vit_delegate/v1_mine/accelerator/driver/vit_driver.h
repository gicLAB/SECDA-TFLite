#ifndef DRIVER_NAME
#define DRIVER_NAME

#include "acc_container.h"

namespace vit_sim {

void block_add(acc_container &drv) {
  prf_start(0);
  // Here we are packing all of the data in, preprocessing and sending it off

  // Get DMA buffer ready for reading

  int i_len = 0;
  int *dma_input_buf =
      drv.mdma->dmas[0].dma_get_inbuffer(); // returns int pointer (32 bit)

  dma_input_buf[i_len++] = drv.pN;
  dma_input_buf[i_len++] = drv.pM;
  dma_input_buf[i_len++] = drv.pK;

  dma_input_buf[i_len++] = drv.crx;
  dma_input_buf[i_len++] = drv.crf;
  dma_input_buf[i_len++] = drv.ra;

  int NO_ROWS = 64;
  int NO_COLS = 64;
  dma_input_buf[i_len++] = NO_ROWS;
  dma_input_buf[i_len++] = NO_COLS;
  drv.mdma->dmas[0].dma_start_send(i_len);
  drv.mdma->dmas[0].dma_wait_send();
  i_len = 0;

  for (int n = 0; n < drv.pN; n += NO_ROWS) {
    int n_remaining = min(NO_ROWS, drv.pN - n);
    dma_input_buf[i_len++] = n_remaining;

    memcpy(dma_input_buf + i_len, drv.padded_input + n * drv.pK,
           drv.pK * n_remaining * sizeof(int8_t));
    i_len += (drv.pK * n_remaining) / 4;

    for (int m = 0; m < drv.pM; m += NO_COLS) {
      int m_remaining = min(NO_COLS, drv.pM - m);
      dma_input_buf[i_len++] = m_remaining;

      memcpy(dma_input_buf + i_len, drv.padded_weights + m * drv.pK,
             drv.pK * m_remaining * sizeof(int8_t));
      i_len += (drv.pK * m_remaining) / 4;

      for (int a = n; a < n + n_remaining; a++) {
        for (int b = m; b < m + m_remaining; b++) {


          dma_input_buf[i_len++] = drv.bias[b] +
                                   (drv.wt_sum[b] * (drv.rhs_offset)) +
                                   (drv.in_sum[a] * (drv.lhs_offset));

        }
      }

      prf_start(0);
      drv.mdma->dmas[0].dma_start_send(i_len);
      drv.mdma->dmas[0].dma_wait_send();
      drv.mdma->dmas[0].dma_start_recv(n_remaining * m_remaining / 4);
      drv.mdma->dmas[0].dma_wait_recv();
      prf_end(0, drv.a_t->fpga_total);

      i_len = 0;
      

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
    // prf_end(2, drv.a_t->wgt_loop);
  }
  // prf_end(1, drv.a_t->inp_loop);
  // prf_end(0, drv.a_t->driver_total);
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
  //
}
} // namespace vit_sim

#endif // DRIVER_NAME