#ifndef DRIVER_NAME
#define DRIVER_NAME

#include "acc_container.h"
namespace driver_name_space {

void BlockTempdel(acc_container &drv) {
  prf_start(1);
  int i_len = 0;
  int *DMA_input_buffer = drv.mdma->dmas[0].dma_get_inbuffer();
  DMA_input_buffer[i_len++] = drv.padded_size / 4;
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

  for (int i = 0; i < drv.padded_size; i += 4) {
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
  prf_end(1, drv.p_t.p_data_copy);

  prf_start(2);
  drv.mdma->dmas[0].dma_start_send(i_len);
  drv.mdma->dmas[0].dma_wait_send();
  prf_end(2, drv.p_t.p_data_send);

  prf_start(3);
  drv.mdma->dmas[0].dma_start_recv(drv.padded_size / 4);
  drv.mdma->dmas[0].dma_wait_recv();
  prf_end(3, drv.p_t.p_compute);

  prf_start(4);
  int8_t *oval =
      reinterpret_cast<int8_t *>(drv.mdma->dmas[0].dma_get_outbuffer());
  for (int i = 0; i < drv.size; i++) {
    drv.output_C[i] = oval[i];
  }
  prf_end(4, drv.p_t.p_data_store);
}

void Entry(acc_container &drv) {
  prf_start(1); // Start profiling the driver
  drv.p_t.t_driver_total = duration_ns::zero(); // Initialize total time
  drv.p_t.t_driver_total += drv.p_t.p_data_copy;
  drv.p_t.t_driver_total += drv.p_t.p_data_send;
  drv.p_t.t_driver_total += drv.p_t.p_compute;
  drv.p_t.t_driver_total += drv.p_t.p_data_store;
#ifdef DELEGATE_VERBOSE
  cout << "===========================" << endl;
  cout << "Tempdel Driver" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "TEMPDEL Layer: " << drv.layer << endl;
  cout << "Input Length: " << drv.padded_size << endl;
  cout << "===========================" << endl;
#endif
  BlockTempdel(drv);
  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.p_t.t_driver_total); // Stop profiling the driver
}

} // namespace driver_name_space
#endif // DRIVER_NAME