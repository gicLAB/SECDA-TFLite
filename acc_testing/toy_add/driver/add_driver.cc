#ifndef ADD_DRIVER
#define ADD_DRIVER

#include "acc_container.h"
// #include "../acc.sc.h"
// #include "sysc_profiler/profiler.h"
// #include "systemc_binding.h"
#include <iostream>

#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_in0 0x16000000
#define dma_out0 0x16800000
#define DMA_BL 4194304

void reset(int *arg0, int *arg1, int *arg2, int size) {
  for (int i = 0; i < size; i++) {
    arg0[i] = i;
    arg1[i] = i;
    arg2[i] = 0;
  }
}

void BlockAdd(acc_container &drv) {
  int i_len = 0;
  int *DMA_input_buffer = drv.sdma->dma_get_inbuffer();
  DMA_input_buffer[i_len++] = drv.length;
  for (int i = 0; i < drv.length; i++) {
    DMA_input_buffer[i_len++] = drv.input_A[i];
    DMA_input_buffer[i_len++] = drv.input_B[i];
  }

  drv.sdma->dma_start_send(i_len);
  drv.sdma->dma_wait_send();
  drv.sdma->dma_start_recv(drv.length);
  drv.sdma->dma_wait_recv();

#ifdef SYSC
  drv.profile->saveProfile(drv.acc->profiling_vars);
#endif
  int *oval = reinterpret_cast<int *>(drv.sdma->dma_get_outbuffer());
  for (int i = 0; i < drv.length; i++) {
    drv.output_C[i] = oval[i];
  }
}

int main() {

  struct acc_container drv;

#ifdef SYSC
  static ACCNAME accelerator("toy_add");
  static struct stream_dma sdma(0, 0, 8096, 0, 8096);
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

  int size = 20;
  auto arg0 = new int[size];
  auto arg1 = new int[size];
  auto arg2 = new int[size];
  reset(arg0, arg1, arg2, size);

  drv.sdma = &sdma;
  drv.length = size;
  drv.input_A = arg0;
  drv.input_B = arg1;
  drv.output_C = arg2;

  std::cout << "Starting BLOCK ADD" << std::endl;
  BlockAdd(drv);
  std::cout << "Ending BLOCK ADD" << std::endl;
  
  // print out the result
  for (int i = 0; i < size; i++) {
    std::cout << arg2[i] << std::endl;
  }

  // profile.saveCSVRecords("toyadd_sim");

  return 0;
}

#endif // ADD_DRIVER