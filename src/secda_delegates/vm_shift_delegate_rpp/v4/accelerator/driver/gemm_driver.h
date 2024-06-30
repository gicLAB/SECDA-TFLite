#ifndef GEMM_DRIVER
#define GEMM_DRIVER

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <sys/stat.h>
#include <typeinfo>

// GEMM_Driver for simulated VM acccelerator
namespace tflite_vm {

void Config_MM(acc_container &drv) {
  int wgt_rows = roundUp(drv.cols, 4);
  int inp_rows = roundUp(drv.rows, 4);
  int rdepth = roundUp(drv.depth, 16);
  drv.mdma->multi_dma_change_start_4(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer();

  int inl0 = 0;
  in0[inl0++] = OPCODE_CONFIG;
  in0[inl0++] = rdepth;
  in0[inl0++] = wgt_rows;
  in0[inl0++] = inp_rows;
  in1[0] = 1;
  in2[0] = 1;
  in3[0] = 1;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(1);
  drv.mdma->dmas[2].dma_start_send(1);
  drv.mdma->dmas[3].dma_start_send(1);
  drv.mdma->dmas[0].dma_wait_send();
  drv.mdma->dmas[1].dma_wait_send();
  drv.mdma->dmas[2].dma_wait_send();
  drv.mdma->dmas[3].dma_wait_send();

}

void ExecuteMM(acc_container &drv) {
  drv.mdma->multi_dma_change_start_4(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  in0[inl0++] = OPCODE_COMPUTE;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[0].dma_wait_send();
  
  drv.mdma->dmas[0].dma_start_recv(1);
  drv.mdma->dmas[1].dma_start_recv(1);
  drv.mdma->dmas[2].dma_start_recv(1);
  drv.mdma->dmas[3].dma_start_recv(1);
  drv.mdma->dmas[0].dma_wait_recv();
  drv.mdma->dmas[1].dma_wait_recv();
  drv.mdma->dmas[2].dma_wait_recv();
  drv.mdma->dmas[3].dma_wait_recv();

}

void TileGEMM(acc_container &drv, int output_stride, int depth, int rdepth,
              int rows, int rrows, int cols, int rcols, int8_t *results) {
  prf_start(1);
  Config_MM(drv);
  ExecuteMM(drv);
  prf_end(1, drv.t2.p_vm_acc);
}

void Entry(acc_container &drv, int8_t *dst) {
  int rows = drv.rows;
  int cols = drv.cols;
  int depth = drv.depth;
  int rrows = roundUp(drv.rows, 2);
  int rcols = roundUp(drv.cols, 4);
  int rdepth = roundUp(drv.depth, 16);
  int output_stride = drv.cols;

#if defined(SYSC) || defined(DELEGATE_VERBOSE)
  // cerr << "VM" << endl;
  // cerr << "===========================" << endl;
  // cerr << "Pre-ACC Info: " << drv.t.layer << endl;
  // cerr << "rdepth: " << rdepth << " depth: " << depth << endl;
  // cerr << "rcols: " << rcols << " cols: " << cols << endl;
  // cerr << "rrows: " << rrows << " rows: " << rows << endl;
  // cerr << "output_stride: " << output_stride << endl;
  // cerr << "===========================" << endl;
#endif

  TileGEMM(drv, output_stride, depth, rdepth, rows, rrows, cols, rcols, dst);
  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/out_vm_" + std::to_string(drv.t.layer) + "_1.csv");
  int8_t *res_pointer = dst;
  int index = 0;
  for (int r = 0; r < rows; r++) {
    myfile << endl;
    for (int c = 0; c < cols; c++) {
      myfile << (int)res_pointer[index] << ",";
      index++;
    }
  }
  myfile.close();
#endif
}

} // namespace tflite_vm
#endif // GEMM_DRIVER
