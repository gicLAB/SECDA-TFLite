#ifndef MM2IM_MT
#define MM2IM_MT

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include <cstring>

#define TOG(X)

namespace mm2im_driver {
using namespace std;

struct Load_Send_Acc : Task {
  Load_Send_Acc(acc_container &drv_, int padded_depth_)
      : drv(drv_), padded_depth(padded_depth_) {}

  void Run() override {
    int starting = 0;
    int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
    for (int o_1 = 0; o_1 < drv.oh; o_1++) {
      TOG(cerr << "Sending rows: " << starting << " to " << drv.oh_ends[o_1] + 1
               << endl;);
      int number_of_rows = drv.oh_ends[o_1] + 1 - starting;
      int inl0 = 0;
      if (drv.oh_ends[o_1] != starting - 1) {
        int padded_depth_4 = padded_depth / 4;
        int inp_packet_a = number_of_rows;
        int inp_packet_b = padded_depth_4 / 4;
        int inp_packet_c = starting;
        int opcode = 4 + 16;
        in0[inl0++] = opcode;
        in0[inl0++] = inp_packet_a;
        in0[inl0++] = inp_packet_b;
        in0[inl0++] = inp_packet_c;
        for (int i = 0; i < number_of_rows; i++) {
          int src_addr = (starting + i) * padded_depth_4;
          memcpy(&in0[inl0], &drv.loaded_inputs[src_addr], padded_depth_4 * 4);
          inl0 += padded_depth_4;
        }
      }
      bool last = (o_1 == drv.oh - 1);
      int schedule = last ? 0 : 16;
      int opcode = 64 + schedule;
      in0[inl0++] = opcode;
      in0[inl0++] = o_1 * drv.ow;
      in0[inl0++] = drv.ow;
      drv.mdma->dmas[0].dma_start_send(inl0);
      drv.mdma->multi_dma_wait_send();
      starting = drv.oh_ends[o_1] + 1;
    }
  }
  acc_container &drv;
  int padded_depth;
};

struct Store_Results_Acc : Task {
  Store_Results_Acc(acc_container &drv_, int o_3_, int filter_step_)
      : drv(drv_), o_3(o_3_), filter_step(filter_step_) {}

  void Run() override {
    int *out0 = drv.mdma->dmas[0].dma_get_outbuffer();
    int8_t *bo0 = reinterpret_cast<int8_t *>(out0);
    for (int o_1 = 0; o_1 < drv.oh; o_1++) {
      drv.mdma->dmas[0].dma_start_recv(drv.ow * filter_step + PE_COUNT + 1);
      drv.mdma->multi_dma_wait_recv();
      int outl0 = 0;
      for (int o_2 = 0; o_2 < drv.ow; o_2++) {
        int o_dex = ((o_1 * drv.ow) + o_2) * drv.oc + o_3;
        memcpy(&drv.output_data[o_dex], &out0[outl0], filter_step);
        outl0 += filter_step / 4;
      }
    }
  }
  acc_container &drv;
  int o_3;
  int filter_step;
};

} // namespace mm2im_driver

#endif // MM2IM_MT