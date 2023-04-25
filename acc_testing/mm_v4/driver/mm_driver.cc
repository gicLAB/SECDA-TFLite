
#include <cstdint>
#include <iostream>

#include "acc_container.h"
#include "mm_man_v4_As.h"
#include "mm_man_v4_Bs.h"
#include "mm_man_v4_Cs.h"
#include "mm_man_v4_Ns.h"

// Problem Size
#ifndef M
#define M 64
#endif

#ifndef N
#define N 64
#endif

#ifndef K
#define K 64
#endif

#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_in0 0x16000000
#define dma_out0 0x16400000
#define DMA_BL 65536

void dump(int *arg0, int *arg1, int *arg2) {
  printf("--\narg0:\n");
  for (int i = 0; i < M; i++) {
    printf("[");
    for (int j = 0; j < K; j++)
      printf("%d,\t", (int)arg0[i * K + j]);
    printf("]\n");
  }
  printf("--\narg1:\n");
  for (int i = 0; i < K; i++) {
    printf("[");
    for (int j = 0; j < N; j++)
      printf("%d,\t", (int)arg1[i * N + j]);
    printf("]\n");
  }
  printf("--\narg2:\n");
  for (int i = 0; i < M; i++) {
    printf("[");
    for (int j = 0; j < N; j++)
      printf("%d,\t", (int)arg2[i * N + j]);
    printf("]\n");
  }
}

void dump_out(int *arg2) {
  printf("--\narg2:\n");
  for (int i = 0; i < M; i++) {
    printf("[");
    for (int j = 0; j < N; j++)
      printf("%d,\t", (int)arg2[i * N + j]);
    printf("]\n");
  }
}

void reset(int *arg0, int *arg1, int *arg2) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      arg0[i * K + j] = i;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      arg1[i * N + j] = j;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      arg2[i * N + j] = 0;
    }
  }
}

void simpleMM(int *arg0, int *arg1, int *arg2) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int acc = 0;
      for (int k = 0; k < K; k++) {
        int x = arg0[m * K + k];
        int y = arg1[k * N + n];
        acc += x * y;
      }
      arg2[m * N + n] = acc;
    }
  }
}

int testCorrect(int *arg1, int *arg2) {
  bool equal = true;
  for (int i = 0; i < N * M; i++) {
    if (arg1[i] != arg2[i]) {
      equal = false;
      break;
    }
  }
  if (!equal)
    std::cout << "  FAILED" << std::endl;
  else
    std::cout << "  PASSED" << std::endl;
  return equal == true ? 0 : -1;
}

int main() {

  struct acc_container drv;
#ifdef SYSC
  static ACCNAME accelerator("toy_add");
  static struct stream_dma sdma(0, 0, 10000, 0, 10000);
  static struct sysC_sigs scs(1);
  Profile profile;
  sysC_init();
  systemC_binder(&accelerator, &sdma, &scs);
  drv.scs = &scs;
  drv.profile = &profile;
  drv.acc = &accelerator;
#else
  int *accelerator = getAccBaseAddress<int>(acc_address, 65536);
  static struct stream_dma sdma(dma_addr0, dma_in0, DMA_BL, dma_out0, DMA_BL);
  drv.acc = accelerator;
#endif

  auto arg0 = new int[M * K];
  auto arg1 = new int[K * N];
  auto arg2 = new int[M * N];
  auto arg3 = new int[M * N];
  reset(arg0, arg1, arg3);
  reset(arg0, arg1, arg2);

  drv.sdma = &sdma;
  drv.A = arg0;
  drv.B = arg1;
  drv.C = arg2;
  drv.M_size = M;
  drv.N_size = N;
  drv.K_size = K;

#if TEST
  // C++ MM implementation
  reset(arg0, arg1, arg3);
  simpleMM(arg0, arg1, arg3);
#endif

  // Acc Version
  reset(arg0, arg1, arg2);
#ifdef ACCv4As
  v4_As(arg0, arg1, arg2);
#elif ACCv4Bs
  v4_Bs(arg0, arg1, arg2);
#elif ACCv4Cs
  v4_Cs(arg0, arg1, arg2);
#else
  v4_Ns(drv);
#endif

  int ret = 0;
#if TEST
  // Compare with C++ MM implementation
  ret = testCorrect(arg2, arg3);
#if DBG
  dump_out(arg3);
#endif
#endif

  delete[] arg0;
  delete[] arg1;
  delete[] arg2;
  delete[] arg3;
  return ret;
}
