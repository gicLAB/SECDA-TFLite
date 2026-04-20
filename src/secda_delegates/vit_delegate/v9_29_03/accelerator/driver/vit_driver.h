#ifndef DRIVER_NAME
#define DRIVER_NAME

#include "acc_container.h"
#include <fstream>
#include <iomanip>
#include <cstring> 
#include <algorithm> 
#include <array>

namespace vit_sim {

// 0: Legacy heuristic (stable default)
// 1: Cost model for all layers
// 2: Cost model for Conv, legacy for FC
#ifndef VIT_MODE_POLICY
#define VIT_MODE_POLICY 0
#endif

namespace {

constexpr int kOutputDmaCount = 4;

inline int CeilDiv(int a, int b) { return (a + b - 1) / b; }

inline int WordsForOutputDma(int total_words, int dma_idx) {
    if (dma_idx >= total_words) {
        return 0;
    }
    return (total_words + kOutputDmaCount - 1 - dma_idx) / kOutputDmaCount;
}

inline void StartOutputDmaRecvs(acc_container &drv, int total_words,
                                                                std::array<int, kOutputDmaCount> &recv_words) {
    for (int d = 0; d < kOutputDmaCount; d++) {
        recv_words[d] = WordsForOutputDma(total_words, d);
        if (recv_words[d] > 0) {
            drv.mdma->dmas[d].dma_start_recv(recv_words[d]);
        }
    }
}

inline void WaitOutputDmaRecvs(acc_container &drv,
                                                             const std::array<int, kOutputDmaCount> &recv_words) {
    for (int d = 0; d < kOutputDmaCount; d++) {
        if (recv_words[d] > 0) {
            drv.mdma->dmas[d].dma_wait_recv();
        }
    }
}

inline void MergeOutputDmaWords(
        acc_container &drv, const std::array<int, kOutputDmaCount> &recv_words,
        int total_words, std::vector<int8_t> &merged_output) {
    merged_output.resize(total_words * 4);

    std::array<const int8_t *, kOutputDmaCount> dma_outputs = {
            reinterpret_cast<const int8_t *>(drv.mdma->dmas[0].dma_get_outbuffer()),
            reinterpret_cast<const int8_t *>(drv.mdma->dmas[1].dma_get_outbuffer()),
            reinterpret_cast<const int8_t *>(drv.mdma->dmas[2].dma_get_outbuffer()),
            reinterpret_cast<const int8_t *>(drv.mdma->dmas[3].dma_get_outbuffer())};

    for (int w = 0; w < total_words; w++) {
        const int d = w & (kOutputDmaCount - 1);
        const int idx_in_dma = w / kOutputDmaCount;
        if (idx_in_dma < recv_words[d]) {
            memcpy(merged_output.data() + (w * 4), dma_outputs[d] + (idx_in_dma * 4),
                         4);
        }
    }
}

inline long long EstimateDenseTransferBytes(const acc_container &drv,
                                            int hw_rows, int hw_cols,
                                            int hw_cores) {
  const int n_tiles = CeilDiv(drv.pN, hw_rows);
  const int m_tiles = CeilDiv(drv.pM, hw_cols * hw_cores);
  const int packed_cols = ((hw_cols + 3) / 4) * 4;

  const long long input_bytes_per_n = static_cast<long long>(hw_rows) * drv.pK;
  const long long weight_bytes_per_m =
      static_cast<long long>(hw_cores) * hw_cols * drv.pK;
  const long long crf_bytes_per_m =
      (drv.layer_t == 1) ? static_cast<long long>(hw_cores) * hw_cols * 4 : 0;
  const long long crx_bytes_per_m =
      (drv.layer_t == 1) ? static_cast<long long>(hw_cores) * packed_cols : 0;
  const long long bias_bytes_per_m =
      drv.is_bias ? static_cast<long long>(hw_cores) * hw_cols * 4 : 0;
  const long long wt_sum_bytes_per_m =
      static_cast<long long>(hw_cores) * hw_cols * 4;
  const long long in_sum_bytes_per_m = static_cast<long long>(hw_rows) * 4;
  const long long offset_bytes_per_m = 8;
  const long long output_bytes_per_m =
      static_cast<long long>(hw_rows) * hw_cols * hw_cores;

  const long long per_m = weight_bytes_per_m + crf_bytes_per_m + crx_bytes_per_m +
                          bias_bytes_per_m + wt_sum_bytes_per_m + in_sum_bytes_per_m +
                          offset_bytes_per_m + output_bytes_per_m;
  return static_cast<long long>(n_tiles) * input_bytes_per_n +
         static_cast<long long>(n_tiles) * m_tiles * per_m;
}

inline long long EstimateMobileTransferBytes(const acc_container &drv,
                                             int hw_rows, int hw_cols,
                                             int hw_cores) {
  const int m_tiles = CeilDiv(drv.pM, hw_cols);
  const int n_tiles = CeilDiv(drv.pN, hw_rows * hw_cores);
  const int packed_cols = ((hw_cols + 3) / 4) * 4;

  const long long weight_bytes_per_m = static_cast<long long>(hw_cols) * drv.pK;
  const long long input_bytes_per_n =
      static_cast<long long>(hw_cores) * hw_rows * drv.pK;
  const long long crf_bytes_per_n =
      (drv.layer_t == 1) ? static_cast<long long>(hw_cols) * 4 : 0;
  const long long crx_bytes_per_n =
      (drv.layer_t == 1) ? static_cast<long long>(packed_cols) : 0;
  const long long bias_bytes_per_n =
      drv.is_bias ? static_cast<long long>(hw_cols) * 4 : 0;
  const long long wt_sum_bytes_per_n = static_cast<long long>(hw_cols) * 4;
  const long long in_sum_bytes_per_n =
      static_cast<long long>(hw_cores) * hw_rows * 4;
  const long long offset_bytes_per_n = 8;
  const long long output_bytes_per_n =
      static_cast<long long>(hw_cores) * hw_rows * hw_cols;

  const long long per_n = input_bytes_per_n + crf_bytes_per_n + crx_bytes_per_n +
                          bias_bytes_per_n + wt_sum_bytes_per_n + in_sum_bytes_per_n +
                          offset_bytes_per_n + output_bytes_per_n;
  return static_cast<long long>(m_tiles) * weight_bytes_per_m +
         static_cast<long long>(m_tiles) * n_tiles * per_n;
}

} // namespace

void block_add(acc_container &drv) {

    // cout << "V8" << endl;

  // Architecture Constants
  const int HW_ROWS = pn_block;  
  const int HW_COLS = pm_block;  
  const int HW_CORES = NUM_CORES;
  const int HW_DENSE_W = HW_COLS * HW_CORES; // 256
  const int HW_MOBILE_IN = HW_ROWS * HW_CORES; // 256

    int mode = (drv.pM >= HW_DENSE_W) ? MODE_DENSE : MODE_MOBILE;

#if VIT_MODE_POLICY == 1 || VIT_MODE_POLICY == 2
    const long long dense_bytes =
            EstimateDenseTransferBytes(drv, HW_ROWS, HW_COLS, HW_CORES);
    const long long mobile_bytes =
            EstimateMobileTransferBytes(drv, HW_ROWS, HW_COLS, HW_CORES);
#endif

#if VIT_MODE_POLICY == 1
    mode = (dense_bytes <= mobile_bytes) ? MODE_DENSE : MODE_MOBILE;
#elif VIT_MODE_POLICY == 2
    if (drv.layer_t != 0) {
        mode = (dense_bytes <= mobile_bytes) ? MODE_DENSE : MODE_MOBILE;
    }
#endif

  int param_len = 0;
  int *param_buf = drv.mdma->dmas[0].dma_get_inbuffer();
  int input_len = 0;
  int *input_buf = drv.mdma->dmas[1].dma_get_inbuffer();
  int weight_len = 0;
  int *weight_buf = drv.mdma->dmas[2].dma_get_inbuffer();
  int bias_len = 0;
  int *bias_buf = drv.mdma->dmas[3].dma_get_inbuffer();
    std::vector<int8_t> merged_output;

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

    const int kBatchWordsWgt = 131072;
    const int kBatchWordsParam = 32768;
    const int kBatchWordsBias = 32768;

    auto flush_send = [&](int dma_idx, int &len) {
        if (len > 0) {
            drv.mdma->dmas[dma_idx].dma_start_send(len);
            len = 0;
        }
    };

    // =========================================================================
    // MODE DENSE
    // =========================================================================
    if (mode == 0) {
        for (int n = 0; n < drv.pN; n += HW_ROWS) {
            prf_start(2);

            int n_rem = std::min(HW_ROWS, drv.pN - n);
            int valid_bytes = drv.pK * n_rem;
            int total_bytes = drv.pK * HW_ROWS;

            memcpy(input_buf + input_len, drv.padded_input + n * drv.pK, valid_bytes);
            if (n_rem < HW_ROWS)
                memset((int8_t *)input_buf + (input_len * 4) + valid_bytes, 0,
                             total_bytes - valid_bytes);
            input_len += total_bytes / 4;

            drv.mdma->dmas[1].dma_start_send(input_len);
            input_len = 0;
            prf_end(2, drv.a_t->inp_copy);

            const int dense_tile_w_words = HW_CORES * (drv.pK * HW_COLS / 4);
            const int dense_tile_param_words =
                    ((drv.layer_t == 1) ? HW_CORES * HW_COLS : 0) + HW_CORES * HW_COLS;
            const int dense_tile_bias_words =
                    2 + ((drv.layer_t == 1) ? HW_CORES * ((HW_COLS + 3) / 4) : 0) +
                    (drv.is_bias ? HW_CORES * HW_COLS : 0) + HW_ROWS;
            const int dense_chunk_tiles = std::max(
                    1, std::min({4,
                                             kBatchWordsWgt / std::max(1, dense_tile_w_words),
                                             kBatchWordsParam / std::max(1, dense_tile_param_words),
                                             kBatchWordsBias / std::max(1, dense_tile_bias_words)}));
              const int dense_chunk_tiles_safe = (drv.layer_t == 1) ? 1 : dense_chunk_tiles;

            for (int m_chunk = 0; m_chunk < drv.pM;
                   m_chunk += dense_chunk_tiles_safe * HW_DENSE_W) {
                const int m_chunk_end =
                    std::min(drv.pM, m_chunk + (dense_chunk_tiles_safe * HW_DENSE_W));

                for (int m = m_chunk; m < m_chunk_end; m += HW_DENSE_W) {
                    for (int c = 0; c < HW_CORES; c++) {
                        int cur_m = m + (c * HW_COLS);
                        int m_rem = (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
                        int w_bytes = drv.pK * HW_COLS;

                        if (m_rem > 0) {
                            int w_valid = drv.pK * m_rem;
                            memcpy(weight_buf + weight_len, drv.padded_weights + cur_m * drv.pK,
                                         w_valid);
                            if (m_rem < HW_COLS)
                                memset((int8_t *)weight_buf + (weight_len * 4) + w_valid, 0,
                                             w_bytes - w_valid);
                        } else {
                            memset((int8_t *)weight_buf + (weight_len * 4), 0, w_bytes);
                        }
                        weight_len += w_bytes / 4;
                    }

                    bias_buf[bias_len++] = drv.rhs_offset;
                    bias_buf[bias_len++] = drv.lhs_offset;

                    if (drv.layer_t == 1) {
                        for (int c = 0; c < HW_CORES; c++) {
                            int cur_m = m + (c * HW_COLS);
                            int m_rem =
                                    (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
                            if (m_rem > 0) {
                                memcpy(param_buf + param_len, drv.crf_a.data() + cur_m, m_rem * 4);
                                if (m_rem < HW_COLS)
                                    memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                            } else {
                                memset(param_buf + param_len, 0, HW_COLS * 4);
                            }
                            param_len += HW_COLS;

                            if (m_rem > 0) {
                                memcpy(bias_buf + bias_len, drv.crx_a.data() + cur_m, m_rem);
                                if (m_rem < HW_COLS)
                                    memset((int8_t *)bias_buf + (bias_len * 4) + m_rem, 0,
                                                 HW_COLS - m_rem);
                            } else {
                                memset((int8_t *)bias_buf + (bias_len * 4), 0, HW_COLS);
                            }
                            bias_len += (HW_COLS + 3) / 4;
                        }
                    }

                    if (drv.is_bias) {
                        for (int c = 0; c < HW_CORES; c++) {
                            int cur_m = m + (c * HW_COLS);
                            int m_rem =
                                    (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;
                            if (m_rem > 0) {
                                memcpy(bias_buf + bias_len, drv.bias + cur_m, m_rem * 4);
                                if (m_rem < HW_COLS)
                                    memset(bias_buf + bias_len + m_rem, 0, (HW_COLS - m_rem) * 4);
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
                            if (m_rem < HW_COLS)
                                memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                        } else {
                            memset(param_buf + param_len, 0, HW_COLS * 4);
                        }
                        param_len += HW_COLS;
                    }

                    memcpy(bias_buf + bias_len, drv.in_sum + n, n_rem * 4);
                    if (n_rem < HW_ROWS)
                        memset(bias_buf + bias_len + n_rem, 0, (HW_ROWS - n_rem) * 4);
                    bias_len += HW_ROWS;
                }

                prf_start(5);
                flush_send(2, weight_len);
                prf_end(5, drv.a_t->wgt_copy);

                prf_start(8);
                flush_send(0, param_len);
                flush_send(3, bias_len);
                drv.mdma->multi_dma_wait_send();
                prf_end(8, drv.a_t->bias_copy);

                for (int m = m_chunk; m < m_chunk_end; m += HW_DENSE_W) {
                    prf_start(9);
                    int recv_size = HW_ROWS * HW_DENSE_W / 4;
                    int8_t *output_val = nullptr;
                    if (drv.layer_t == 0) {
                        std::array<int, kOutputDmaCount> recv_words = {0, 0, 0, 0};
                        StartOutputDmaRecvs(drv, recv_size, recv_words);
                        WaitOutputDmaRecvs(drv, recv_words);
                        MergeOutputDmaWords(drv, recv_words, recv_size, merged_output);
                        output_val = merged_output.data();
                    } else {
                        drv.mdma->dmas[0].dma_start_recv(recv_size);
                        drv.mdma->dmas[0].dma_wait_recv();
                        output_val = (int8_t *)drv.mdma->dmas[0].dma_get_outbuffer();
                    }

                    for (int i = 0; i < n_rem; i++) {
                        int row_base = n * drv.pM + m;
                        int row_offset_in_output = i * drv.pM;

                        for (int c = 0; c < HW_CORES; c++) {
                            int cur_m = m + (c * HW_COLS);
                            int m_rem =
                                    (cur_m < drv.pM) ? std::min(HW_COLS, drv.pM - cur_m) : 0;

                            if (m_rem > 0) {
                                int src_offset = (c * HW_ROWS * HW_COLS) + (i * HW_COLS);
                                memcpy(drv.padded_output +
                                                     (row_base + row_offset_in_output + (c * HW_COLS)),
                                             output_val + src_offset, m_rem);
                            }
                        }
                    }
                    prf_end(9, drv.a_t->out_copy);
                }
            }
        }
    }
  
    // =========================================================================
    // MODE MOBILE
    // =========================================================================
    else {
        for (int m = 0; m < drv.pM; m += HW_COLS) {
            int m_rem = std::min(HW_COLS, drv.pM - m);

            prf_start(5);
            int w_bytes = drv.pK * HW_COLS;
            memcpy(weight_buf + weight_len, drv.padded_weights + m * drv.pK,
                         drv.pK * m_rem);
            if (m_rem < HW_COLS)
                memset((int8_t *)weight_buf + (weight_len * 4) + (drv.pK * m_rem), 0,
                             drv.pK * (HW_COLS - m_rem));
            weight_len += w_bytes / 4;
            drv.mdma->dmas[2].dma_start_send(weight_len);
            weight_len = 0;
            prf_end(5, drv.a_t->wgt_copy);

            const int mobile_tile_input_words = HW_CORES * (drv.pK * HW_ROWS / 4);
            const int mobile_tile_param_words =
                    ((drv.layer_t == 1) ? HW_COLS : 0) + HW_COLS;
            const int mobile_tile_bias_words =
                    2 + ((drv.layer_t == 1) ? ((HW_COLS + 3) / 4) : 0) +
                    (drv.is_bias ? HW_COLS : 0) + HW_CORES * HW_ROWS;
            const int mobile_chunk_tiles = std::max(
                    1, std::min({4,
                                             kBatchWordsWgt / std::max(1, mobile_tile_input_words),
                                             kBatchWordsParam / std::max(1, mobile_tile_param_words),
                                             kBatchWordsBias / std::max(1, mobile_tile_bias_words)}));
              const int mobile_chunk_tiles_safe = (drv.layer_t == 1) ? 1 : mobile_chunk_tiles;

            for (int n_chunk = 0; n_chunk < drv.pN;
                   n_chunk += mobile_chunk_tiles_safe * HW_MOBILE_IN) {
                const int n_chunk_end =
                    std::min(drv.pN, n_chunk + (mobile_chunk_tiles_safe * HW_MOBILE_IN));

                for (int n = n_chunk; n < n_chunk_end; n += HW_MOBILE_IN) {
                    for (int c = 0; c < HW_CORES; c++) {
                        int cur_n = n + (c * HW_ROWS);
                        int n_sub = (cur_n < drv.pN) ? std::min(HW_ROWS, drv.pN - cur_n) : 0;
                        int i_bytes = drv.pK * HW_ROWS;

                        if (n_sub > 0) {
                            memcpy(input_buf + input_len, drv.padded_input + cur_n * drv.pK,
                                         drv.pK * n_sub);
                            if (n_sub < HW_ROWS)
                                memset((int8_t *)input_buf + (input_len * 4) + (drv.pK * n_sub), 0,
                                             drv.pK * (HW_ROWS - n_sub));
                        } else {
                            memset((int8_t *)input_buf + (input_len * 4), 0, i_bytes);
                        }
                        input_len += i_bytes / 4;
                    }

                    bias_buf[bias_len++] = drv.rhs_offset;
                    bias_buf[bias_len++] = drv.lhs_offset;

                    if (drv.layer_t == 1) {
                        memcpy(param_buf + param_len, drv.crf_a.data() + m, m_rem * 4);
                        if (m_rem < HW_COLS)
                            memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                        param_len += HW_COLS;
                    }

                    memcpy(param_buf + param_len, drv.wt_sum + m, m_rem * 4);
                    if (m_rem < HW_COLS)
                        memset(param_buf + param_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                    param_len += HW_COLS;

                    if (drv.layer_t == 1) {
                        memcpy(bias_buf + bias_len, drv.crx_a.data() + m, m_rem);
                        if (m_rem < HW_COLS)
                            memset((int8_t *)bias_buf + (bias_len * 4) + m_rem, 0,
                                         HW_COLS - m_rem);
                        bias_len += (HW_COLS + 3) / 4;
                    }

                    if (drv.is_bias) {
                        memcpy(bias_buf + bias_len, drv.bias + m, m_rem * 4);
                        if (m_rem < HW_COLS)
                            memset(bias_buf + bias_len + m_rem, 0, (HW_COLS - m_rem) * 4);
                        bias_len += HW_COLS;
                    }

                    for (int c = 0; c < HW_CORES; c++) {
                        int cur_n = n + (c * HW_ROWS);
                        int n_sub = (cur_n < drv.pN) ? std::min(HW_ROWS, drv.pN - cur_n) : 0;
                        memcpy(bias_buf + bias_len, drv.in_sum + cur_n, n_sub * 4);
                        if (n_sub < HW_ROWS)
                            memset(bias_buf + bias_len + n_sub, 0, (HW_ROWS - n_sub) * 4);
                        bias_len += HW_ROWS;
                    }
                }

                prf_start(2);
                flush_send(1, input_len);
                prf_end(2, drv.a_t->inp_copy);

                prf_start(8);
                flush_send(0, param_len);
                flush_send(3, bias_len);
                drv.mdma->multi_dma_wait_send();
                prf_end(8, drv.a_t->bias_copy);

                for (int n = n_chunk; n < n_chunk_end; n += HW_MOBILE_IN) {
                    prf_start(9);
                    int recv_size = HW_MOBILE_IN * HW_COLS / 4;
                    int8_t *output_val = nullptr;
                    if (drv.layer_t == 0) {
                        std::array<int, kOutputDmaCount> recv_words = {0, 0, 0, 0};
                        StartOutputDmaRecvs(drv, recv_size, recv_words);
                        WaitOutputDmaRecvs(drv, recv_words);
                        MergeOutputDmaWords(drv, recv_words, recv_size, merged_output);
                        output_val = merged_output.data();
                    } else {
                        drv.mdma->dmas[0].dma_start_recv(recv_size);
                        drv.mdma->dmas[0].dma_wait_recv();
                        output_val = (int8_t *)drv.mdma->dmas[0].dma_get_outbuffer();
                    }
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
                            if (n_sub < HW_ROWS)
                                out_buf_offset += (HW_ROWS - n_sub) * HW_COLS;
                        } else {
                            out_buf_offset += HW_ROWS * HW_COLS;
                        }
                    }
                    prf_end(9, drv.a_t->out_copy);
                }
            }
        }
    }

}

void Entry(acc_container &drv) {
  block_add(drv);
}
} 
#endif