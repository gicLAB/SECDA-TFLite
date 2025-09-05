#ifndef DRIVER_NAME
#define DRIVER_NAME

// CORRECT

#include "acc_container.h"

namespace vit_sim {

void block_add(acc_container &drv) {

  // cout << "int is " << sizeof(int) << endl; // 4 bits

  int i_len_1 = 0;
  // dib = dibfer
  int *dib_1 =
      drv.mdma->dmas[0].dma_get_inbuffer(); // returns int pointer (32 bit)

  int i_len_2 = 0;
  int *dib_2 =
      drv.mdma->dmas[1].dma_get_inbuffer();

  int i_len_3 = 0;
  int *dib_3 =
      drv.mdma->dmas[2].dma_get_inbuffer(); 

  int i_len_4 = 0;
  int *dib_4 =
      drv.mdma->dmas[3].dma_get_inbuffer(); 


  dib_1[i_len_1++] = drv.pN;
  dib_1[i_len_1++] = drv.pM;
  dib_1[i_len_1++] = drv.pK;

  dib_1[i_len_1++] = drv.crx;
  dib_1[i_len_1++] = drv.crf;
  dib_1[i_len_1++] = drv.ra;

  int NO_ROWS = 64;
  int NO_COLS = 64;


  dib_1[i_len_1++] = NO_ROWS;
  dib_1[i_len_1++] = NO_COLS;

//   cout << "(V2) Layer info: " << "PN: " << drv.pN << " PM: " << drv.pM
//        << " PK: " << drv.pK << " CRX: " << drv.crx << " CRF: " << drv.crf
//        << " RA: " << drv.ra << " NO_ROWS: " << NO_ROWS
//        << " NO_COLS: " << NO_COLS << endl;

  // cout << "ViT_ACC (V2)" << endl;

  drv.mdma->dmas[0].dma_start_send(i_len_1);
  drv.mdma->dmas[0].dma_wait_send();

  i_len_1 = 0;

  for (int n = 0; n < drv.pN; n += NO_ROWS) {
    int n_remaining = min(NO_COLS, drv.pN - n);
    int n_chunk_size = drv.pK * (n_remaining / 4); 

    dib_1[i_len_1++] = n_remaining;

    // memcpy(dest, src, size) 
    // Copies size bytes from src to dest

    // sizeof(int8_t) used to be at the bottom (n_chunk_size * sizeof(int8_t) and 
    // multiplying n_chunk sized but it equals 1 so I removed it)
    memcpy(dib_1 + i_len_1, drv.padded_input + n * drv.pK,
           n_chunk_size);
    i_len_1 += n_chunk_size / 4; //4 as 1 i_len++ is sending 4 values

    memcpy(dib_2 + i_len_2,
           drv.padded_input + n * drv.pK + n_chunk_size,
           n_chunk_size);
    i_len_2 += n_chunk_size / 4;

    memcpy(dib_3 + i_len_3,
           drv.padded_input + n * drv.pK + 2 * n_chunk_size,
           n_chunk_size);
    i_len_3 += n_chunk_size / 4;

    memcpy(dib_4 + i_len_4,
           drv.padded_input + n * drv.pK + 3 * n_chunk_size,
           n_chunk_size);
    i_len_4 += n_chunk_size / 4;

    drv.mdma->dmas[0].dma_start_send(i_len_1);
    drv.mdma->dmas[1].dma_start_send(i_len_2);
    drv.mdma->dmas[2].dma_start_send(i_len_3);
    drv.mdma->dmas[3].dma_start_send(i_len_4);
    drv.mdma->multi_dma_wait_send();


    i_len_1 = 0;
    i_len_2 = 0;
    i_len_3 = 0;
    i_len_4 = 0;

    for (int m = 0; m < drv.pM; m += NO_COLS) {
      int m_remaining = min(NO_COLS, drv.pM - m);
      int m_chunk_size = drv.pK * (m_remaining / 4);
      dib_1[i_len_1++] = m_remaining;

      memcpy(dib_1 + i_len_1, drv.padded_weights + m * drv.pK,
             m_chunk_size);
      i_len_1 += m_chunk_size / 4;

      memcpy(dib_2 + i_len_2,
             drv.padded_weights + m * drv.pK + m_chunk_size,
             m_chunk_size);
      i_len_2 += m_chunk_size / 4;

      memcpy(dib_3 + i_len_3,
             drv.padded_weights + m * drv.pK +
                 2 * m_chunk_size,
             m_chunk_size);
      i_len_3 += m_chunk_size / 4;

      memcpy(dib_4 + i_len_4,
             drv.padded_weights + m * drv.pK +
                 3 * m_chunk_size,
             m_chunk_size);
      i_len_4 += m_chunk_size / 4;

      drv.mdma->dmas[0].dma_start_send(i_len_1);
      drv.mdma->dmas[1].dma_start_send(i_len_2);
      drv.mdma->dmas[2].dma_start_send(i_len_3);
      drv.mdma->dmas[3].dma_start_send(i_len_4);
      drv.mdma->multi_dma_wait_send();

      i_len_1 = 0;
      i_len_2 = 0;
      i_len_3 = 0;
      i_len_4 = 0;

 
      for (int a = n; a < n + n_remaining; a+=4) {
        for (int b = m; b < m + m_remaining; b++) {
       //    cout << "(v2)bias: " << drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
       //                       (drv.in_sum[a] * (drv.lhs_offset)) << endl;
              dib_1[i_len_1++] = drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
                             (drv.in_sum[a + 0] * (drv.lhs_offset));

              dib_2[i_len_2++] = drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
                             (drv.in_sum[a + 1] * (drv.lhs_offset));

              dib_3[i_len_3++] = drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
                             (drv.in_sum[a + 2] * (drv.lhs_offset));

              dib_4[i_len_4++] = drv.bias[b] + (drv.wt_sum[b] * (drv.rhs_offset)) +
                             (drv.in_sum[a + 3] * (drv.lhs_offset));
        }
      }

      prf_start(0); // fpga_total TODO: Chagne thsi prf

      
      drv.mdma->dmas[0].dma_start_send(i_len_1);
      drv.mdma->dmas[1].dma_start_send(i_len_2);
      drv.mdma->dmas[2].dma_start_send(i_len_3);
      drv.mdma->dmas[3].dma_start_send(i_len_4);
      drv.mdma->multi_dma_wait_send();
      drv.mdma->dmas[0].dma_start_recv(n_remaining * m_remaining /
                                       4);
       drv.mdma->dmas[0].dma_wait_recv();

       // drv.mdma->dmas[0].dma_start_recv(n_remaining * m_remaining / 16); // / ?
       // drv.mdma->dmas[1].dma_start_recv(n_remaining * m_remaining / 16);
       // drv.mdma->dmas[2].dma_start_recv(n_remaining * m_remaining / 16);
       // drv.mdma->dmas[3].dma_start_recv(n_remaining * m_remaining / 16);
                                       
//       drv.mdma->multi_dma_wait_recv();
      prf_end(0, drv.a_t->fpga_total); // fpga_total

      i_len_1 = 0;
      i_len_2 = 0;
      i_len_3 = 0;
      i_len_4 = 0;

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