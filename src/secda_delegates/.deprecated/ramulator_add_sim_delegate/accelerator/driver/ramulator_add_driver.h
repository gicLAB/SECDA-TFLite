#ifndef RAMULATOR_ADD_DRIVER
#define RAMULATOR_ADD_DRIVER

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
namespace tflite_ramulator_addsim {
// Is it possible to get latency number for all memory access within this file,
// if so we can use this to add to the latency of the driver. Additionally if
// can generate a dummy binary which contains the instructions between memory
// instructions we could use this inform ramulator this information to get more
// accurate latecy numbers.

// Also we should run SystemC simulation for main memory accesses.
// We should create dummy main memory controller which generate read and write
// traces.
// We should have generation and simulation mode for the driver as well.

void BlockRamulator_add(acc_container &drv) {
  int i_len = 0;
  int *DMA_input_buffer = drv.sdma->dma_get_inbuffer();
  // DMA_input_buffer[i_len++] = drv.length / 4;
  // DMA_input_buffer[i_len++] = drv.lshift;
  // DMA_input_buffer[i_len++] = drv.in1_off;
  // DMA_input_buffer[i_len++] = drv.in1_sv;
  // DMA_input_buffer[i_len++] = drv.in1_mul;
  // DMA_input_buffer[i_len++] = drv.in2_off;
  // DMA_input_buffer[i_len++] = drv.in2_sv;
  // DMA_input_buffer[i_len++] = drv.in2_mul;
  // DMA_input_buffer[i_len++] = drv.out1_off;
  // DMA_input_buffer[i_len++] = drv.out1_sv;
  // DMA_input_buffer[i_len++] = drv.out1_mul;
  // DMA_input_buffer[i_len++] = drv.qa_max;
  // DMA_input_buffer[i_len++] = drv.qa_min;
  drv.massign<int>(DMA_input_buffer, &drv.length, i_len++, 0, drv.length / 4);
  drv.massign<int>(DMA_input_buffer, &drv.lshift, i_len++, 0, drv.lshift);
  drv.massign<int>(DMA_input_buffer, &drv.in1_off, i_len++, 0, drv.in1_off);
  drv.massign<int>(DMA_input_buffer, &drv.in1_sv, i_len++, 0, drv.in1_sv);
  drv.massign<int>(DMA_input_buffer, &drv.in1_mul, i_len++, 0, drv.in1_mul);
  drv.massign<int>(DMA_input_buffer, &drv.in2_off, i_len++, 0, drv.in2_off);
  drv.massign<int>(DMA_input_buffer, &drv.in2_sv, i_len++, 0, drv.in2_sv);
  drv.massign<int>(DMA_input_buffer, &drv.in2_mul, i_len++, 0, drv.in2_mul);
  drv.massign<int>(DMA_input_buffer, &drv.out1_off, i_len++, 0, drv.out1_off);
  drv.massign<int>(DMA_input_buffer, &drv.out1_sv, i_len++, 0, drv.out1_sv);
  drv.massign<int>(DMA_input_buffer, &drv.out1_mul, i_len++, 0, drv.out1_mul);
  drv.massign<int>(DMA_input_buffer, &drv.qa_max, i_len++, 0, drv.qa_max);
  drv.massign<int>(DMA_input_buffer, &drv.qa_min, i_len++, 0, drv.qa_min);

  // Break input send into 2 loads
  // drv.sdma->dma_start_send(i_len);
  // drv.sdma->dma_wait_send();
  // i_len = 0;

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
    b_val[3] = drv.input_B[i + 3];

    // DMA_input_buffer[i_len++] = aval[0];
    // DMA_input_buffer[i_len++] = bval[0];
    drv.massign<int>(DMA_input_buffer, &aval[0], i_len++, 0, aval[0]);
    drv.massign<int>(DMA_input_buffer, &bval[0], i_len++, 0, bval[0]);
  }

  drv.sdma->dma_start_send(i_len);
  drv.sdma->dma_wait_send();
  drv.sdma->dma_start_recv(drv.length / 4);
  drv.sdma->dma_wait_recv();

  drv.profile->saveProfile(drv.acc->profiling_vars);
  int8_t *oval = reinterpret_cast<int8_t *>(drv.sdma->dma_get_outbuffer());
  for (int i = 0; i < drv.length; i++) {
    // drv.output_C[i] = oval[i];
    drv.massign<int8_t>(drv.output_C, &oval[i], i, 0, oval[i]);
  }
}

void Entry(acc_container &drv) {
#ifdef DELEGATE_VERBOSE
  cout << "Ramulator_addRamulator_add" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "RAMULATOR_ADD Layer: " << drv.layer << endl;
  cout << "Input Length: " << drv.length << endl;
  cout << "===========================" << endl;
#endif


  drv.load_inject_dram_cycles();
  BlockRamulator_add(drv);
}

} // namespace tflite_ramulator_addsim
#endif // RAMULATOR_ADD_DRIVER