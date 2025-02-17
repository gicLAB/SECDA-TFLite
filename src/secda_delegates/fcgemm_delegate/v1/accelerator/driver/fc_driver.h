#ifndef FC_DRIVER
#define FC_DRIVER

#include "acc_container.h"
#include "secda_tools/secda_utils/utils.h"

// FC_Driver for FC-GEMM acccelerator
namespace tflite_fcgemm {

void createWeightLoad(unsigned long long *insn, int &idx, int wgt_start,
                      int depth, int m_inc) {
  int doffset = wgt_start * (depth / 8);
  int dstride = (depth / 8);
  int x_size = (depth / 8);
  int y_size = m_inc;

  unsigned long long p1 = 0;
  unsigned long long p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 1;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createInputLoad(unsigned long long *insn, int &idx, int inp_start,
                     int depth, int n_inc) {
  int doffset = inp_start * (depth / 8);
  int dstride = (depth / 8);
  int x_size = (depth / 8);
  int y_size = n_inc;

  unsigned long long p1 = 0;
  unsigned long long p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 2;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createBiasLoad(unsigned long long *insn, int &idx, int bias_start,
                    int stride, int n_inc, int m_inc) {
  int doffset = bias_start / 2;
  int dstride = stride / 2;
  int x_size = n_inc / 2;
  int y_size = m_inc;

  unsigned long long p1 = 0;
  unsigned long long p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 3;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createCompute(unsigned long long *insn, int &idx, int out_start,
                   int stride, int inp_block, int wgt_block) {
  int doffset = out_start / 4;
  int dstride = stride / 4;
  int x_size = wgt_block;
  int y_size = inp_block;

  unsigned long long p1 = 0;
  unsigned long long p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 0;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void BlockFC(acc_container &drv) {
  int inp_max = INP_SIZE;
  int wgt_max = WGT_SIZE;
  int acc_max = ACC_SIZE;
  int k_inc = drv.pK;
  int m_inc = min((wgt_max), drv.pM);
  int n_inc = min((inp_max), drv.pN);

  while ((n_inc * k_inc > inp_max) && n_inc != 16) n_inc -= 16;
  while ((m_inc * k_inc > wgt_max) && m_inc != 16) m_inc -= 16;
  while ((n_inc * m_inc > acc_max) && n_inc != 16) n_inc -= 16;
  while ((n_inc * m_inc > acc_max) && m_inc != 16) m_inc -= 16;

  // Create 2D Biases
  int32_t *wt_sum = drv.wt_sum;
  int32_t *in_sum = drv.in_sum;
#ifndef SYSC
  int32_t *bias_buf = (int32_t *)(drv.bias_mem);
#else
  int32_t *bias_buf = new int32_t[drv.pN * drv.pM];
#endif
  prf_start(0);
  create_2d_biases(0, drv.pN, 0, drv.pM, bias_buf, drv.bias, wt_sum, in_sum,
                   drv.rhs_offset, drv.lhs_offset, drv.K);
  prf_end(0, drv.t2.p_bpack);

  // Create Instructions
  int insn_idx = 0;
#ifndef SYSC
  unsigned long long *insn_mem = drv.insn_mem;
#else
  unsigned int insn_count_exp = (roundUp(drv.pK, k_inc) / k_inc) *
                                (roundUp(drv.pM, m_inc) / m_inc) *
                                (2 + (roundUp(drv.pN, n_inc) / n_inc) * 6);
  unsigned int insn_count = 0;
  for (int k = 0; k < drv.pK; k += k_inc) { // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    for (int m = 0; m < drv.pM; m += m_inc) { // Weight Dim
      int m_b = min(m_inc, drv.pM - m);
      insn_count += 2;
      for (int n = 0; n < drv.pN; n += n_inc) { // Input Dim
        int n_b = min(n_inc, drv.pN - n);
        insn_count += 6;
      }
    }
  }
  if (insn_count_exp != insn_count) {
    cout << "Error in Instruction Count" << endl;
    cout << "Expected: " << insn_count_exp << " Got: " << insn_count << endl;
  }
  unsigned long long *insn =
      (unsigned long long *)malloc(sizeof(unsigned long long) * insn_count);
#endif
  for (int k = 0; k < drv.pK; k += k_inc) { // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    for (int m = 0; m < drv.pM; m += m_inc) { // Weight Dim
      int m_b = min(m_inc, drv.pM - m);
      // Load Weight
      createWeightLoad(insn, insn_idx, m, drv.pK, m_b);
      for (int n = 0; n < drv.pN; n += n_inc) { // Input Dim
        int n_b = min(n_inc, drv.pN - n);
        createInputLoad(insn, insn_idx, n, drv.pK, n_b);
        createBiasLoad(insn, insn_idx, drv.pN * m + n, drv.pN, n_b, m_b);
        createCompute(insn, insn_idx, drv.pM * n + m, drv.pM, n_b, m_b);
      }
    }
  }
// Move Instructions to MMapped DMA buffer to enable accelerator access
#ifndef SYSC
  writeMappedReg<int>(drv.acc, 0x5c, drv.crf);
  writeMappedReg<int>(drv.acc, 0x64, drv.crx);
  writeMappedReg<int>(drv.acc, 0x6c, drv.ra);
  writeMappedReg<int>(drv.acc, 0x74, drv.pK);
  writeMappedReg<int>(drv.acc, 0x2c, insn_idx / 2);
  writeMappedReg<int>(drv.acc, 0x14, ++drv.start_count);
  // Start Accelerator
  prf_start(1);
  bool done = readMappedReg<int>(drv.acc, 0x1c) == drv.start_count;

  // Wait for Accelerator to finish
  while (!done) {
    done = readMappedReg<int>(drv.acc, 0x1c) == drv.start_count;
  }
  prf_end(1, drv.t2.acc);
#else
  drv.scs->sig_crf = drv.crf;
  drv.scs->sig_crx = drv.crx;
  drv.scs->sig_ra = drv.ra;
  drv.scs->sig_depth = drv.pK;
  drv.scs->sig_insn_count = insn_idx / 2;
  drv.scs->sig_start_acc = ++drv.start_count;

  // Move Input data to MMapped DMA buffer to enable accelerator access
  drv.scs->insn_mem.burst_write(0, insn_idx, insn);
  drv.scs->inp_mem.burst_write(0, drv.pN * drv.pK / 8,
                               (unsigned long long *)&drv.padded_input[0]);
  drv.scs->wgt_mem.burst_write(0, drv.pM * drv.pK / 8,
                               (unsigned long long *)&drv.padded_weights[0]);

  drv.scs->bias_mem.burst_write(0, drv.pN * drv.pM / 2,
                                (unsigned long long *)&bias_buf[0]);
  // Start Accelerator Simulation
  sc_start();
  // drv.profile->saveProfile(drv.acc->profiling_vars);
  // Retrive Output data from  MMapped DMA buffer
  unsigned int *out_set = (unsigned int *)drv.padded_output;
  int out_len = drv.pN * drv.pM / 4;
  drv.scs->out_mem.burst_read(0, out_len, out_set);
#endif

  prf_start(2);
  store_unpad(drv.padded_output, drv.N, drv.M, drv.output_data);
  prf_end(2, drv.t2.p_unpack);
}

void Entry(acc_container &drv) {
  // #ifdef DELEGATE_VERBOSE
  cout << "FC ACC - Layer: " << drv.t.layer << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "padded_K: " << drv.pK << " K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << " M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << " N: " << drv.N << endl;
  cout << "===========================" << endl;
  // #endif
  BlockFC(drv);
  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
}
} // namespace tflite_fcgemm
#endif // FC_DRIVER

// for (int k = 0; k < drv.pK; k += k_inc) { // Common Dim
//   int k_b = min(k_inc, drv.pK - k);
//   for (int m = 0; m < drv.pM; m += m_inc) { // Weight Dim
//     int m_b = min(m_inc, drv.pM - m);
//     insn_count += 2;
//     for (int n = 0; n < drv.pN; n += n_inc) { // Input Dim
//       int n_b = min(n_inc, drv.pN - n);
//       insn_count += 6;
//     }
//   }
// }