// Vector Add Experiment

#include <fstream>
#include <iostream>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif

#include "accelerator/driver/driver.h"
#include "secda_tools/secda_profiler/profiler.h"

unsigned int dma_addrs[1] = {dma_addr0};
unsigned int dma_addrs_in[1] = {dma_in0};
unsigned int dma_addrs_out[1] = {dma_out0};
struct acc_times a_t;
static struct Profile profile;

#define DELLOG(X) X

#ifdef SYSC
ACCNAME *acc;
struct sysC_sigs *scs;
struct s_mdma *mdma;
#else
int *acc;
struct s_mdma *mdma;
#endif

struct a_ctrl *ctrl;
static h_ctrl *hwc;

using namespace std;

int main() {
  // ========================================
  // ========================================
  // Initialize the Accelerator

  DELLOG(std::cout << "===========================" << std::endl;);
#ifdef SYSC
  static ACCNAME _acc("ACCNAME");
  static struct sysC_sigs scs1(1);
  static struct a_ctrl ctrl1;
  static struct h_ctrl hwc1;
  static struct s_mdma mdma1(1, dma_addrs, dma_addrs_in, dma_addrs_out,
                             DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);

  sysC_init();
  hwc1.init_hwc(HWC_Monitor_Count);
  ctrl1.init_sigs(CTRL_Reg_Count);

  sysC_binder(&_acc, &scs1, &ctrl1, &hwc1, &mdma1);
  acc = &_acc;
  scs = &scs1;
  ctrl = &ctrl1;
  hwc = &hwc1;
  mdma = &mdma1;
  DELLOG(std::cout << "Initialised the SystemC Modules" << std::endl;);

#else
  acc = getAccBaseAddress<int>(acc_ctrl_address, 65536);
  int *acc_ctrl_base = getAccBaseAddress<int>(acc_ctrl_address, 65536);
  int *acc_hwc_base = getAccBaseAddress<int>(acc_hwc_address, 65536);
  static struct a_ctrl ctrl1(acc_ctrl_base);
  static struct h_ctrl hwc1(acc_hwc_base);
  static struct s_mdma mdma1(1, dma_addrs, dma_addrs_in, dma_addrs_out,
                             DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE);
  ctrl1.init_sigs(CTRL_Reg_Count);
  hwc1.init_hwc(HWC_Monitor_Count);
  ctrl = &ctrl1;
  hwc = &hwc1;
  mdma = &mdma1;
  DELLOG(std::cout << "Initialised the DMA" << std::endl;);
#endif
  DELLOG(std::cout << "VADD_ACC Accelerator";);
  DELLOG(std::cout << std::endl;);
  DELLOG(std::cout << "===========================" << std::endl;);

  // ========================================
  // ========================================
  // Define problem parameters
  int L = VEC_LENGTH; // Vector length

  std::vector<int> X_vec(L);
  std::vector<int> Y_vec(L);
  std::vector<int> Z_vec(L, 0);

  // Initialize input vectors
  for (int i = 0; i < L; i++) {
    X_vec[i] = i;
    Y_vec[i] = i * 2;
  }

  // ========================================
  // ========================================
  // FPGA Impl
  acc_container drv;

#ifdef SYSC
  drv.scs = scs;
#endif
  drv.profile = &profile;
  drv.acc = acc;
  drv.ctrl = ctrl;
  drv.a_t = &a_t;
  drv.hwc = hwc;
  drv.mdma = mdma;

  drv.L = L;

  drv.X = X_vec.data();
  drv.Y = Y_vec.data();
  drv.Z = Z_vec.data();

  // Call FPGA driver
  prf_start(1);
  acc_driver::Entry(drv);
  prf_end(1, a_t.fpga_total);
  DELLOG(cout << "FPGA Done!" << endl;);

  // Verify against CPU reference
  bool ok = true;
  for (int i = 0; i < L; i++) {
    int expected = X_vec[i] + Y_vec[i];
    if (Z_vec[i] != expected) {
      ok = false;
      std::cout << "Mismatch at index " << i << ": expected " << expected
                << ", got " << Z_vec[i] << std::endl;
      break;
    }else {
      DELLOG(std::cout << "Index " << i << ": expected " << expected
                       << ", got " << Z_vec[i] << std::endl;);
    }
  }
  std::cout << (ok ? "Validation: PASSED\n" : "Validation: FAILED\n");
  a_t.print();

  int return_code = ok ? 0 : 1;
  return return_code;
}
