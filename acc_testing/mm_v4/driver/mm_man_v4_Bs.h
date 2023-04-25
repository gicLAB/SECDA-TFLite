#ifndef MM_MAN_v4_BS_H
#define MM_MAN_v4_BS_H

#include "acc_container.h"
#include <iostream>

#define A_buffer 4096
#define B_buffer 4096
#define C_buffer 4096

#define tile_M 16
#define tile_N 16
#define tile_K 16

void v4_Bs(acc_container &drv) {
  int M = drv.M_size;
  int N = drv.N_size;
  int K = drv.K_size;

#ifndef block_N
  // Block N: tiling factor for dim N, after taking into account the size of A
  // and C buffers, and compute tile size for M and K
  int block_N = std::min(C_buffer / tile_M, std::min(B_buffer / tile_K, N));
#endif

#ifndef block_M
  // Block M: tiling factor for dim M, after taking to account size of B and C
  // buffers, Block K and N, and compute tile size for K
  int block_M = std::min(C_buffer / block_N, std::min(A_buffer / tile_K, M));
#endif

#ifndef block_K
  // Block K: tiling factor for dim K, after taking into account: size of A and
  // B buffers, and compute tile size for N and M
  int block_K = std::min(B_buffer / block_M, std::min(A_buffer / block_N, K));
#endif

  // Gets pointer to DMA_IN_BUFFER
  int *dma_inbuffer = drv.sdma->dma_get_inbuffer();
  // Gets pointer to DMA_OUT_BUFFER
  int *dma_outbuffer = drv.sdma->dma_get_outbuffer();

  // Start Tiling
  for (int n = 0; n < N; n += block_N) {
    for (int k = 0; k < K; k += block_K) {

      // Data_len is used to track what is in the DMA_IN_BUFFER
      int data_len = 0;

      // Encodes HEADER; Tells accelerator to expect A, B tiles and compute C
      uint32_t op_code = 2;
      uint32_t ce_a = 0;
      ce_a += block_K;
      ce_a = ce_a << 10;
      ce_a += block_N;
      ce_a = ce_a << 10;
      ce_a += block_M;

      dma_inbuffer[data_len++] = op_code;
      dma_inbuffer[data_len++] = ce_a;

      // Copies B into DMA_IN_BUFFER; Increments data_len by length of B
      for (int tk = 0; tk < block_K; tk++)
        for (int tn = 0; tn < block_N; tn++)
          dma_inbuffer[data_len + block_N * tk + tn] =
              drv.B[(k + tk) * N + (n + tn)];
      data_len += block_K * block_N;

      // Sends data_len of data
      drv.sdma->dma_start_send(data_len);
      drv.sdma->dma_wait_send();

      for (int m = 0; m < M; m += block_M) {

        data_len = 0;
        // Encodes HEADER; Tells accelerator to expect send C
        uint32_t op_code = 13;
        uint32_t ce_a = 0;
        ce_a += block_K;
        ce_a = ce_a << 10;
        ce_a += block_N;
        ce_a = ce_a << 10;
        ce_a += block_M;

        dma_inbuffer[data_len++] = op_code;
        dma_inbuffer[data_len++] = ce_a;

        // Copies A into DMA_IN_BUFFER; Increments data_len by length of A
        for (int tm = 0; tm < block_M; tm++)
          for (int tk = 0; tk < block_K; tk++)
            dma_inbuffer[data_len + block_K * tm + tk] =
                drv.A[(m + tm) * K + (k + tk)];
        data_len += block_M * block_K;

        drv.sdma->dma_start_send(data_len);
        drv.sdma->dma_wait_send();

        // Indicates to DMA, how much space is available and where it is
        drv.sdma->dma_start_recv(block_M * block_N);
        drv.sdma->dma_wait_recv();

        // Copies result from DMA_OUT_BUFFER to padded output buffer
        for (int tm = 0; tm < block_M; tm++) {
          for (int tn = 0; tn < block_N; tn++) {
            drv.C[(m + tm) * N + n + tn] += dma_outbuffer[block_N * tm + tn];
          }
        }
      }
    }
  }
  drv.sdma->dma_free();
}

#endif /* MM_MAN_v4_BS_H */