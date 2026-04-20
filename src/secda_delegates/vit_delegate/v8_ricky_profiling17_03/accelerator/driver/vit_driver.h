#ifndef DRIVER_NAME
#define DRIVER_NAME

#include "acc_container.h"
#include <fstream>
#include <iomanip>
#include <cstring> 
#include <algorithm> 

namespace vit_sim {

void block_add(acc_container &drv) {

    // cout << "V8" << endl;

  // Architecture Constants
  const int HW_ROWS = pn_block;  
  const int HW_COLS = pm_block;  
  const int HW_CORES = NUM_CORES;
  const int HW_DENSE_W = HW_COLS * HW_CORES; // 256
  const int HW_MOBILE_IN = HW_ROWS * HW_CORES; // 256
  const int CORE_BLOCK_SIZE = HW_ROWS * HW_COLS; // 4096 bytes per core block

  int mode = (drv.pM >= HW_DENSE_W) ? 0 : 1; 

  int param_len = 0;
  int *param_buf = drv.mdma->dmas[0].dma_get_inbuffer();
  int input_len = 0;
  int *input_buf = drv.mdma->dmas[1].dma_get_inbuffer();
  int weight_len = 0;
  int *weight_buf = drv.mdma->dmas[2].dma_get_inbuffer();
  int bias_len = 0;
  int *bias_buf = drv.mdma->dmas[3].dma_get_inbuffer();

  prf_start(1); 

  // Header
  param_buf[param_len++] = drv.layer_t;
  param_buf[param_len++] = mode;
  param_buf[param_len++] = drv.pN;
  param_buf[param_len++] = drv.pM;
  param_buf[param_len++] = drv.pK;
//   cout << "pN: " << drv.pN << " pM: " << drv.pM << " pK: " << drv.pK << endl;
  param_buf[param_len++] = drv.ra;
  param_buf[param_len++] = drv.is_bias;
  param_buf[param_len++] = HW_ROWS;
  param_buf[param_len++] = HW_COLS;

  if (drv.layer_t == 0) {
      param_buf[param_len++] = drv.crf; 
      param_buf[param_len++] = drv.crx; 
  }

  drv.mdma->dmas[0].dma_start_send(param_len); param_len = 0;
  drv.mdma->dmas[0].dma_wait_send();
  prf_end(1, drv.a_t->param_copy);

  // =========================================================================
  // MODE DENSE
  // =========================================================================
  if (mode == 0) { 
    for (int n = 0; n < drv.pN; n += HW_ROWS) {
      prf_start(2);

      // 1. Inputs (Shared)
      int n_rem = std::min(HW_ROWS, drv.pN - n);
      int valid_bytes = drv.pK * n_rem;
      int total_bytes = drv.pK * HW_ROWS;
      
      memcpy(input_buf + input_len, drv.padded_input + n * drv.pK, valid_bytes);
      if (n_rem < HW_ROWS) memset((int8_t*)input_buf + (input_len*4) + valid_bytes, 0, total_bytes - valid_bytes);
      input_len += total_bytes / 4;
      
      drv.mdma->dmas[1].dma_start_send(input_len); input_len = 0;
      prf_end(2, drv.a_t->inp_copy);

      for (int m = 0; m < drv.pM; m += HW_DENSE_W) {
        prf_start(5);
        // 2. Weights (Unique)
        for (int c = 0; c < HW_CORES; c++) {
            int cur_m = m + (c * HW_COLS);
            int m_rem = (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
            int w_bytes = drv.pK * HW_COLS;

            if (m_rem > 0) {
                int w_valid = drv.pK * m_rem;
                memcpy(weight_buf + weight_len, drv.padded_weights + cur_m * drv.pK, w_valid);
                if (m_rem < HW_COLS) memset((int8_t*)weight_buf + (weight_len*4) + w_valid, 0, w_bytes - w_valid);
            } else {
                memset((int8_t*)weight_buf + (weight_len*4), 0, w_bytes);
            }
            weight_len += w_bytes / 4;
        }
        drv.mdma->dmas[2].dma_start_send(weight_len); weight_len = 0;
        prf_end(5, drv.a_t->wgt_copy);

        prf_start(8);
        bias_buf[bias_len++] = drv.rhs_offset; bias_buf[bias_len++] = drv.lhs_offset;

        // 3. Params
        prf_start(1);
        if (drv.layer_t == 1) {
            for (int c = 0; c < HW_CORES; c++) {
                 int cur_m = m + (c * HW_COLS);
                 int m_rem = (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
                 if (m_rem > 0) {
                     memcpy(param_buf + param_len, drv.crf_a.data() + cur_m, m_rem * 4);
                     if (m_rem < HW_COLS) memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                 } else {
                     memset(param_buf + param_len, 0, HW_COLS * 4);
                 }
                 param_len += HW_COLS; 

                 if (m_rem > 0) {
                      memcpy(bias_buf + bias_len, drv.crx_a.data() + cur_m, m_rem);
                      if (m_rem < HW_COLS) memset((int8_t*)bias_buf + (bias_len*4) + m_rem, 0, HW_COLS - m_rem);
                 } else {
                      memset((int8_t*)bias_buf + (bias_len*4), 0, HW_COLS); 
                 }
                 bias_len += (HW_COLS + 3) / 4;
            }
        }

        if (drv.is_bias) {
             for (int c = 0; c < HW_CORES; c++) {
                 int cur_m = m + (c * HW_COLS);
                 int m_rem = (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
                 if (m_rem > 0) {
                     memcpy(bias_buf + bias_len, drv.bias + cur_m, m_rem * 4);
                     if (m_rem < HW_COLS) memset(bias_buf + bias_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                 } else {
                     memset(bias_buf + bias_len, 0, HW_COLS * 4);
                 }
                 bias_len += HW_COLS;
             }
        }

        for (int c = 0; c < HW_CORES; c++) {
             int cur_m = m + (c * HW_COLS);
             int m_rem = (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
             if (m_rem > 0) {
                 memcpy(param_buf + param_len, drv.wt_sum + cur_m, m_rem * 4);
                 if (m_rem < HW_COLS) memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
             } else {
                 memset(param_buf + param_len, 0, HW_COLS * 4);
             }
             param_len += HW_COLS;
        }
        drv.mdma->dmas[0].dma_start_send(param_len); param_len = 0;
        prf_end(1, drv.a_t->param_copy);

        memcpy(bias_buf + bias_len, drv.in_sum + n, n_rem * 4);
        if (n_rem < HW_ROWS) memset(bias_buf + bias_len + n_rem, 0, (HW_ROWS - n_rem) * 4);
        bias_len += HW_ROWS;

        drv.mdma->dmas[3].dma_start_send(bias_len); bias_len = 0;
        prf_end(8, drv.a_t->bias_copy);
        drv.mdma->multi_dma_wait_send();
        
        // --- OUTPUTS (FIXED UNSHUFFLE LOGIC) ---
        prf_start(9);
        int recv_size = HW_ROWS * HW_DENSE_W / 4; 
        drv.mdma->dmas[0].dma_start_recv(recv_size);
        drv.mdma->dmas[0].dma_wait_recv();

        int8_t *output_val = (int8_t*)drv.mdma->dmas[0].dma_get_outbuffer();
            
        // Hardware sends: [Core0 64x64] [Core1 64x64] [Core2 64x64] [Core3 64x64]
        // We need to reconstruct linear rows.
        
        for (int i = 0; i < n_rem; i++) { // For each Row (0..63)
          int row_base = n * drv.pM + m; 
          int row_offset_in_output = i * drv.pM; // Destination Row Start
          
          for (int c = 0; c < HW_CORES; c++) {
              int cur_m = m + (c * HW_COLS);
              int m_rem = (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
              
              if (m_rem > 0) {
                  // Source: Core c's block, Row i
                  int src_offset = (c * CORE_BLOCK_SIZE) + (i * HW_COLS);
                  
                  // Dest: Global Output
                  // Destination = (Base + RowOffset) + (Core * 64)
                  memcpy(drv.padded_output + (row_base + row_offset_in_output + (c * HW_COLS)),
                         output_val + src_offset, 
                         m_rem);
              }
          }
        }
        prf_end(9, drv.a_t->out_copy);
      }
    }
  } 
  
  // =========================================================================
  // MODE MOBILE
  // =========================================================================
  else { 
    for (int m = 0; m < drv.pM; m += HW_COLS) {
        prf_start(5);

        // 1. Weights
        int m_rem = std::min(HW_COLS, drv.pM - m);
        int w_bytes = drv.pK * HW_COLS;
        
        memcpy(weight_buf + weight_len, drv.padded_weights + m * drv.pK, drv.pK * m_rem);
        if (m_rem < HW_COLS) memset((int8_t*)weight_buf + (weight_len*4) + (drv.pK*m_rem), 0, drv.pK*(HW_COLS - m_rem));
        weight_len += w_bytes / 4;
        
        drv.mdma->dmas[2].dma_start_send(weight_len); weight_len = 0;
        prf_end(5, drv.a_t->wgt_copy);

        for (int n = 0; n < drv.pN; n += HW_MOBILE_IN) {
            prf_start(2);

            // 2. Inputs
            for (int c = 0; c < HW_CORES; c++) {
                int cur_n = n + (c * HW_ROWS);
                int n_sub = (cur_n < drv.pN) ? std::min(HW_ROWS, drv.pN - cur_n) : 0;
                int i_bytes = drv.pK * HW_ROWS;

                if (n_sub > 0) {
                    memcpy(input_buf + input_len, drv.padded_input + cur_n * drv.pK, drv.pK * n_sub);
                    if (n_sub < HW_ROWS) memset((int8_t*)input_buf + (input_len * 4) + (drv.pK * n_sub), 0, drv.pK * (HW_ROWS - n_sub));
                } else {
                    memset((int8_t*)input_buf + (input_len * 4), 0, i_bytes);
                }
                input_len += i_bytes / 4;
            }
            drv.mdma->dmas[1].dma_start_send(input_len); input_len = 0;
            prf_end(2, drv.a_t->inp_copy);

            prf_start(8);
            bias_buf[bias_len++] = drv.rhs_offset; bias_buf[bias_len++] = drv.lhs_offset;

            // 3. Params
            prf_start(1);
            if (drv.layer_t == 1) {
                memcpy(param_buf + param_len, drv.crf_a.data() + m, m_rem * 4);
                if (m_rem < HW_COLS) memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                param_len += HW_COLS;
            }

            memcpy(param_buf + param_len, drv.wt_sum + m, m_rem * 4);
            if (m_rem < HW_COLS) memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
            param_len += HW_COLS;
            drv.mdma->dmas[0].dma_start_send(param_len); param_len = 0;
            prf_end(1, drv.a_t->param_copy);

            if (drv.layer_t == 1) {
                memcpy(bias_buf + bias_len, drv.crx_a.data() + m, m_rem);
                if (m_rem < HW_COLS) memset((int8_t*)bias_buf + (bias_len*4) + m_rem, 0, HW_COLS - m_rem);
                bias_len += (HW_COLS + 3) / 4;
            }

            if (drv.is_bias) {
                memcpy(bias_buf + bias_len, drv.bias + m, m_rem * 4);
                if (m_rem < HW_COLS) memset(bias_buf + bias_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                bias_len += HW_COLS;
            }

            for (int c = 0; c < HW_CORES; c++) {
                 int cur_n = n + (c * HW_ROWS);
                 int n_sub = (cur_n < drv.pN) ? std::min(HW_ROWS, drv.pN - cur_n) : 0;
                 memcpy(bias_buf + bias_len, drv.in_sum + cur_n, n_sub * 4);
                 if (n_sub < HW_ROWS) memset(bias_buf + bias_len + n_sub, 0, (HW_ROWS - n_sub) * 4);
                 bias_len += HW_ROWS;
            }

            drv.mdma->dmas[3].dma_start_send(bias_len); bias_len = 0;
            prf_end(8, drv.a_t->bias_copy);
            drv.mdma->multi_dma_wait_send();

            // Output
            prf_start(9);
            int recv_size = HW_MOBILE_IN * HW_COLS / 4; 
            drv.mdma->dmas[0].dma_start_recv(recv_size);
            drv.mdma->dmas[0].dma_wait_recv();

            int8_t *output_val = (int8_t*)drv.mdma->dmas[0].dma_get_outbuffer();
            
            // Scatter (Unpad)
            // HW sends Core-Major: [Core0 Block] [Core1 Block] ...
            // This is correct for Mobile Mode output scatter logic.
            int out_buf_offset = 0;
            for (int c = 0; c < HW_CORES; c++) {
                int cur_n = n + (c * HW_ROWS);
                int n_sub = (cur_n < drv.pN) ? std::min(HW_ROWS, drv.pN - cur_n) : 0;
                
                if (n_sub > 0) {
                    for (int i = 0; i < n_sub; i++) {
                        int base = (cur_n + i) * drv.pM + m; 
                        memcpy(drv.padded_output + base, output_val + out_buf_offset, m_rem);
                        out_buf_offset += HW_COLS; 
                    }
                    if (n_sub < HW_ROWS) out_buf_offset += (HW_ROWS - n_sub) * HW_COLS;
                } else {
                    out_buf_offset += HW_ROWS * HW_COLS; 
                }
            }
            prf_end(9, drv.a_t->out_copy);
        }
    }
  }
}

void Entry(acc_container &drv) {
  block_add(drv);
}
} 
#endif