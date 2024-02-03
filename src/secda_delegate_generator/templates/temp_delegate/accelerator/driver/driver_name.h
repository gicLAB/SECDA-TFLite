#ifndef DRIVER_NAME
#define DRIVER_NAME

#include "acc_container.h"
#include "secda_tflite_path/threading_utils/utils.h"
namespace tflite_tempsim {

void BlockTemp(acc_container &drv) {
  int i_len = 0;
  int *DMA_input_buffer = drv.mdma->dmas[0].dma_get_inbuffer();
  DMA_input_buffer[i_len++] = drv.length / 4;
  DMA_input_buffer[i_len++] = drv.lshift;
  DMA_input_buffer[i_len++] = drv.in1_off;
  DMA_input_buffer[i_len++] = drv.in1_sv;
  DMA_input_buffer[i_len++] = drv.in1_mul;
  DMA_input_buffer[i_len++] = drv.in2_off;
  DMA_input_buffer[i_len++] = drv.in2_sv;
  DMA_input_buffer[i_len++] = drv.in2_mul;
  DMA_input_buffer[i_len++] = drv.out1_off;
  DMA_input_buffer[i_len++] = drv.out1_sv;
  DMA_input_buffer[i_len++] = drv.out1_mul;
  DMA_input_buffer[i_len++] = drv.qa_max;
  DMA_input_buffer[i_len++] = drv.qa_min;
  int8_t a_val[4];
  int8_t b_val[4];
  int *aval = reinterpret_cast<int *>(a_val);
  int *bval = reinterpret_cast<int *>(b_val);

  for (int i = 0; i < drv.length; i += 4) {
    a_val[0] = drv.input_A[i + 0];
    a_val[1] = drv.input_A[i + 1];
    a_val[2] = drv.input_A[i + 2];
    a_val[3] = drv.input_A[i + 3];
    b_val[0] = drv.input_B[i + 0];
    b_val[1] = drv.input_B[i + 1];
    b_val[2] = drv.input_B[i + 2];
    b_val[3] = drv.input_B[i + 3];
    DMA_input_buffer[i_len++] = aval[0];
    DMA_input_buffer[i_len++] = bval[0];
  }

  drv.mdma->dmas[0].dma_start_send(i_len);
  drv.mdma->dmas[0].dma_wait_send();
  drv.mdma->dmas[0].dma_start_recv(drv.length / 4);
  drv.mdma->dmas[0].dma_wait_recv();

  int8_t *oval =
      reinterpret_cast<int8_t *>(drv.mdma->dmas[0].dma_get_outbuffer());
  for (int i = 0; i < drv.length; i++) {
    drv.output_C[i] = oval[i];
  }
}

void Entry(acc_container &drv) {
  prf_start(1); // Start profiling the driver
#ifdef DELEGATE_VERBOSE
  cout << "TempTemp" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "TEMP Layer: " << drv.layer << endl;
  cout << "Input Length: " << drv.length << endl;
  cout << "===========================" << endl;
#endif
  BlockTemp(drv);
  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.p_t.driver_total); // Stop profiling the driver
}

} // namespace tflite_tempsim
#endif // DRIVER_NAME