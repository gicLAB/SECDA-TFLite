#ifndef DRIVER_NAME
#define DRIVER_NAME

#include "acc_container.h"

namespace vit_sim {

void block_add(acc_container &drv) {

  int *param_buf = drv.mdma->dmas[0].dma_get_inbuffer();
  int *input_buf = drv.mdma->dmas[1].dma_get_inbuffer();
  int *weight_buf = drv.mdma->dmas[2].dma_get_inbuffer();
  int *bias_buf = drv.mdma->dmas[3].dma_get_inbuffer();

  // --- Send Global Parameters (once per layer) ---
  int param_len = 0;
  prf_start(1); // param_copy
  param_buf[param_len++] = drv.pN;
  param_buf[param_len++] = drv.pM;
  param_buf[param_len++] = drv.pK;
  param_buf[param_len++] = drv.crx;
  param_buf[param_len++] = drv.crf;
  param_buf[param_len++] = drv.ra;
  param_buf[param_len++] = drv.is_bias;
  prf_end(1, drv.a_t->param_copy);
  drv.mdma->dmas[0].dma_start_send(param_len);
  // We wait here to ensure global params are set before tiles are sent
  drv.mdma->dmas[0].dma_wait_send(); 

  //! Tile over N dimension
  for (int n = 0; n < drv.pN; n += pn_tile) {
    int n_remaining = min(pn_tile, drv.pN - n);

    //! Tile over M dimension
    for (int m = 0; m < drv.pM; m += pm_tile) {
      int m_remaining = min(pm_tile, drv.pM - m);

      //! Tile over K dimension
      for (int k = 0; k < drv.pK; k += pk_tile) {
        int k_remaining = min(pk_tile, drv.pK - k);
        bool is_first_k = (k == 0);
        bool is_last_k = ((k + pk_tile) >= drv.pK);

        // --- Send tile-specific parameters ---
        // FIX: We send these on the same channel as other params (0)
        // and do NOT block/wait here.
        param_len = 0;
        param_buf[param_len++] = n_remaining;
        param_buf[param_len++] = m_remaining;
        param_buf[param_len++] = k_remaining;
        param_buf[param_len++] = is_first_k;
        param_buf[param_len++] = is_last_k;
        drv.mdma->dmas[0].dma_start_send(param_len);
        // REMOVED: dma_wait_send() was here, causing a bottleneck.

        // --- Send Input Tile ---
        prf_start(2); // input_copy
        int input_len_bytes = 0;
        for (int i = 0; i < n_remaining; ++i) { // Where is this buffer?
          memcpy((int8_t *)input_buf + input_len_bytes,
                 drv.padded_input + (n + i) * drv.pK + k,
                 k_remaining * sizeof(int8_t));
          input_len_bytes += k_remaining;
        }
        drv.mdma->dmas[1].dma_start_send((input_len_bytes + 3) / 4);
        prf_end(2, drv.a_t->inp_copy);

        // --- Send Weight Tile ---
        prf_start(3); // weight_copy
        int weight_len_bytes = 0;
        for (int i = 0; i < m_remaining; ++i) {
          memcpy((int8_t *)weight_buf + weight_len_bytes,
                 drv.padded_weights + (m + i) * drv.pK + k,
                 k_remaining * sizeof(int8_t));
          weight_len_bytes += k_remaining;
        }
        drv.mdma->dmas[2].dma_start_send((weight_len_bytes + 3) / 4);
        prf_end(3, drv.a_t->wgt_copy);

        // --- Send Bias/Sum Data (only on the last k-tile) ---
        if (is_last_k) {
          prf_start(8); // bias_copy
          int bias_len = 0;
          bias_buf[bias_len++] = drv.rhs_offset;
          bias_buf[bias_len++] = drv.lhs_offset;
          memcpy(bias_buf + bias_len, drv.bias + m, m_remaining * sizeof(int));
          bias_len += m_remaining;
          
          // FIX: Consolidate wt_sum and in_sum into the BIAS stream (channel 3)
          // This is the key change to prevent deadlock.
          memcpy(bias_buf + bias_len, drv.wt_sum + m, m_remaining * sizeof(int));
          bias_len += m_remaining;
          memcpy(bias_buf + bias_len, drv.in_sum + n, n_remaining * sizeof(int));
          bias_len += n_remaining;
          
          drv.mdma->dmas[3].dma_start_send(bias_len);
          prf_end(8, drv.a_t->bias_copy);
        }

        // --- Synchronize all sends for this tile ---
        // This is the single wait point for all data transfers for the current k-tile.
        drv.mdma->multi_dma_wait_send();

        // --- Receive result (only after the last k-tile) ---
        if (is_last_k) {
          drv.mdma->dmas[0].dma_start_recv(n_remaining * m_remaining / 4);
          drv.mdma->dmas[0].dma_wait_recv();

          int8_t *output_val =
              reinterpret_cast<int8_t *>(drv.mdma->dmas[0].dma_get_outbuffer());
          for (int i = 0; i < n_remaining; i++) {
            int base = (n + i) * drv.pM + m;
            int output_offset = i * m_remaining;
            prf_start(9); // out_copy
            memcpy(drv.padded_output + base, output_val + output_offset,
                   m_remaining * sizeof(int8_t));
            prf_end(9, drv.a_t->out_copy);
          }
        }
      }
    }
  }
}

void Entry(acc_container &drv) {
#ifdef DELEGATE_VERBOSE
  cout << "FC ACC - Layer: " << drv.layer << endl;
  cout << "===============";
  cout << "Pre-ACC Info" << endl;
  cout << "padded_K: " << drv.pK << "K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << "M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << "N: " << drv.N << endl;
  cout << "===========================";
#endif

  block_add(drv);
}
} // namespace vit_sim

#endif // DRIVER_NAME
