
#ifndef ACC_DRIVER
#define ACC_DRIVER

#include "acc_container.h"

#define DLOG(X)

namespace acc_driver {

void ACC_Offload(acc_container &drv) {

  // Problem specific parameters
  int L = drv.L; // Vector length

  int *X = drv.X;
  int *Y = drv.Y;
  int *Z = drv.Z;

  drv.hwc->set_target_state(0, 3);
  drv.hwc->set_target_state(1, 2);
  drv.hwc->set_target_state(2, 2);
  drv.hwc->reset_hwc();

  // Gets pointer to DMA_IN_BUFFER
  int *dma_inbuffer = drv.mdma->dmas[0].dma_get_inbuffer();

  // Data_len is used to track what is in the DMA_IN_BUFFER
  int data_len = 0;

  // Encodes HEADER; Tells accelerator to load X, Y, compute Z, and send Z
  uint32_t op_code = 15; // load_X | load_Y | compute_Z | send_Z
  uint32_t ce_l = L;
  dma_inbuffer[data_len++] = op_code;
  dma_inbuffer[data_len++] = ce_l;

  // Copies X into DMA_IN_BUFFER
  for (int i = 0; i < L; i++)
    dma_inbuffer[data_len++] = X[i];

  // Copies Y into DMA_IN_BUFFER
  for (int i = 0; i < L; i++)
    dma_inbuffer[data_len++] = Y[i];

  // Sends data_len of data
  drv.mdma->dmas[0].dma_start_send(data_len);

  // Waits for data to transfer to finish
  drv.mdma->dmas[0].dma_wait_send();

  // Indicates to DMA, how much space is available and where it is
  drv.mdma->dmas[0].dma_start_recv(L);

  // Waits for data to be received (including TLAST signal)
  drv.mdma->dmas[0].dma_wait_recv();

  // Gets pointer to DMA_OUT_BUFFER
  int *dma_outbuffer = drv.mdma->dmas[0].dma_get_outbuffer();

  // Copies result from DMA_OUT_BUFFER to output vector Z
  for (int i = 0; i < L; i++) {
    Z[i] = dma_outbuffer[i];
  }

  drv.hwc->print_hwc_map(true);
  drv.ctrl->print_reg_map(true);
}

void Entry(acc_container &drv) { ACC_Offload(drv); }

} // namespace acc_driver

#endif // ACC_DRIVER
