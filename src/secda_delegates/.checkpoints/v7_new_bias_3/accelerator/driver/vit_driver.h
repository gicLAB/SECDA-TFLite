#ifndef DRIVER_NAME
#define DRIVER_NAME

// DOCKER

#include "acc_container.h"
#include <fstream>
#include <iomanip>

namespace vit_sim {

// Function to log padded_output to a text file for debugging
void log_padded_output_to_file(
    acc_container &drv, const std::string &filename_prefix = "vit_output") {
  static int call_count = 0;
  call_count++;

  std::string filename = "debugging/" + filename_prefix + "_layer" +
                         std::to_string(drv.layer) + "_call" +
                         std::to_string(call_count) + ".txt";

  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }

  outfile << "VIT Delegate Output Debug Log\n";
  outfile << "============================\n";
  outfile << "Layer: " << drv.layer << "\n";
  outfile << "Call: " << call_count << "\n";
  outfile << "Dimensions: N=" << drv.N << ", M=" << drv.M << ", K=" << drv.K
          << "\n";
  outfile << "Padded Dimensions: pN=" << drv.pN << ", pM=" << drv.pM
          << ", pK=" << drv.pK << "\n";
  outfile << "Output size (pN * pM): " << (drv.pN * drv.pM) << "\n";
  outfile << "============================\n\n";

  // Write the padded_output data
  outfile << "Padded Output Data (int8_t values):\n";
  for (int n = 0; n < drv.pN; n++) {
    outfile << "Row " << n << ": ";
    for (int m = 0; m < drv.pM; m++) {
      int index = n * drv.pM + m;
      outfile << std::setw(4) << static_cast<int>(drv.padded_output[index]);
      if (m < drv.pM - 1) outfile << ", ";
    }
    outfile << "\n";
  }

  outfile.close();
  std::cout << "Debug: Logged padded_output to " << filename << std::endl;
}

void block_add(acc_container &drv) {

  int param_len = 0;
  int *param_buf = drv.mdma->dmas[0].dma_get_inbuffer();

  int input_len = 0;
  int *input_buf = drv.mdma->dmas[1].dma_get_inbuffer();

  int weight_len = 0;
  int *weight_buf = drv.mdma->dmas[2].dma_get_inbuffer();

  int bias_len = 0;
  int *bias_buf = drv.mdma->dmas[3].dma_get_inbuffer();

  prf_start(1); // param_copy

  param_buf[param_len++] = drv.layer_t;

  param_buf[param_len++] = drv.pN;
  param_buf[param_len++] = drv.pM;
  param_buf[param_len++] = drv.pK;

  param_buf[param_len++] = drv.ra;

  param_buf[param_len++] = drv.is_bias;

  int NO_ROWS = pnF;
  int NO_COLS = pmF;

  param_buf[param_len++] = NO_ROWS;
  param_buf[param_len++] = NO_COLS;

  if (drv.layer_t == 0) {
    int crf =
        drv.crf; // Assuming all values are the same for padding calculation
    param_buf[param_len++] = crf;
    int crx =
        drv.crx; // Assuming all values are the same for padding calculation
    param_buf[param_len++] = crx;
  }

  drv.mdma->dmas[0].dma_start_send(param_len);
  param_len = 0;
  drv.mdma->dmas[0].dma_wait_send();

  for (int n = 0; n < drv.pN; n += NO_ROWS) {
    int n_remaining = min(NO_COLS, drv.pN - n);
    input_buf[input_len++] = n_remaining;

    prf_start(2); // input_copy

    // memcpy (dest, src, size)
    memcpy(input_buf + input_len, drv.padded_input + n * drv.pK,
           drv.pK * n_remaining * sizeof(int8_t));
    input_len += (drv.pK * n_remaining) / 4;
    prf_end(2, drv.a_t->inp_copy);

    drv.mdma->dmas[1].dma_start_send(input_len);
    input_len = 0;

    for (int m = 0; m < drv.pM; m += NO_COLS) {

      int m_remaining = min(NO_COLS, drv.pM - m);
      weight_buf[weight_len++] = m_remaining;

      // prf_start(3);
      memcpy(weight_buf + weight_len, drv.padded_weights + m * drv.pK,
             drv.pK * m_remaining * sizeof(int8_t));
      weight_len += (drv.pK * m_remaining) / 4;
      // prf_end(3, drv.a_t->wgt_copy); // weight_copy

      drv.mdma->dmas[2].dma_start_send(weight_len);
      weight_len = 0;

      prf_start(8);
      bias_buf[bias_len++] = drv.rhs_offset;
      bias_buf[bias_len++] = drv.lhs_offset;

      if (drv.layer_t == 1) {

        memcpy(param_buf + param_len, drv.crf_a.data() + m,
               m_remaining * sizeof(int));
        param_len += m_remaining;

        memcpy(bias_buf + bias_len, drv.crx_a.data() + m,
               m_remaining * sizeof(int8_t));
        // bias_len += (m_remaining + 3) / 4; // ! CHANGED
        bias_len += m_remaining / 4;
      }
      if (drv.is_bias == 1) {
        memcpy(bias_buf + bias_len, drv.bias + m, m_remaining * sizeof(int));
        bias_len += m_remaining;
      }
      memcpy(param_buf + param_len, drv.wt_sum + m, m_remaining * sizeof(int));
      param_len += m_remaining;
      memcpy(bias_buf + bias_len, drv.in_sum + n, n_remaining * sizeof(int));
      bias_len += n_remaining;
      prf_end(8, drv.a_t->bias_copy);

      drv.mdma->dmas[0].dma_start_send(param_len);
      param_len = 0;
      drv.mdma->dmas[3].dma_start_send(bias_len);
      bias_len = 0;

      drv.mdma->multi_dma_wait_send();
      drv.mdma->dmas[0].dma_start_recv(n_remaining * m_remaining / 4);
      drv.mdma->dmas[0].dma_wait_recv();

      int8_t *output_val =
          reinterpret_cast<int8_t *>(drv.mdma->dmas[0].dma_get_outbuffer());
      for (int i = 0; i < n_remaining; i++) {
        int base = n * drv.pM + m;
        int offset = i * drv.pM;
        int output_offset = i * m_remaining;

        prf_start(9); // out_copy
        memcpy(drv.padded_output + (base + offset), output_val + output_offset,
               m_remaining * sizeof(int8_t));
        prf_end(9, drv.a_t->out_copy);
      }
    }
  }

  // Log the padded_output to file for debugging
  // log_padded_output_to_file(drv);
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