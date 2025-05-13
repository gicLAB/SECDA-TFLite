#ifndef MM2IM_DRIVER
#define MM2IM_DRIVER

#include "acc_container.h"
#include "mm2im_mt.h"
#include "mm2im_util.h"
#include <assert.h>
#include <cstring>
#include <iostream>

// #define TOG(X)                                                                 \
//   if (drv.verb) X;

#define TOG(X)

int input_data_sent = 0;
int weight_data_sent = 0;
int colmap_data_sent = 0;
int inp_load_calls = 0;
int wgt_load_calls = 0;
int colmap_load_calls = 0;
int data_transfered = 0;
int write_data_recv = 0;

namespace mm2im_driver {
using namespace std;

Load_Send_Acc *LSA;
Store_Results_Acc *SRA;

void LoadConfig(acc_container &drv, int padded_depth) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int opcode = 1;
  in0[inl0++] = 1;
  in0[inl0++] = padded_depth / UF;
  in0[inl0++] = drv.ow;
  in0[inl0++] = (drv.rows / drv.f);
  in0[inl0++] = drv.cols;
  in0[inl0++] = drv.ra;
  in0[inl0++] = drv.oh;

  // pattern opt
  in0[inl0++] = drv.ow;
  in0[inl0++] = drv.ks;
  in0[inl0++] = drv.sx;
  in0[inl0++] = drv.sy;
  in0[inl0++] = drv.pt;
  in0[inl0++] = drv.pl;
  in0[inl0++] = drv.width_col;

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->multi_dma_wait_send();
  prf_end(0, drv.p_t.p_load_config);
}

void LoadWeight(acc_container &drv, int starting_row, int number_of_rows,
                int padded_depth, int filter_step, int starting_filter) {
  prf_start(0);

  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int opcode = 2;
  int wgt_packet_a = number_of_rows;
  int wgt_packet_b = padded_depth_4 / (UF / 4);

  drv.weight_preloaded = drv.preloader->load_weights(drv.mdma);
  if (drv.weight_preloaded) {
    prf_end(0, drv.p_t.p_load_wgt);
    return;
  }

  in0[inl0++] = opcode;
  in0[inl0++] = wgt_packet_a;
  in0[inl0++] = wgt_packet_b;
  in0[inl0++] = filter_step;
  TOG(cerr << "Starting Row: " << starting_row << " Number of Rows: "
           << number_of_rows << " Padded Depth: " << padded_depth
           << " Filter Step: " << filter_step
           << " Starting Filter: " << starting_filter << endl;);
  for (int i = 0; i < number_of_rows; i++) {
    int src_addr = (starting_row + i) * padded_depth_4;
    memcpy(&in0[inl0], &drv.loaded_weights[src_addr], padded_depth_4 * 4);
    inl0 += padded_depth_4;
    in0[inl0++] = drv.acc_wt_sum[starting_row + i] * drv.rhs_offset;
  }
  memcpy(&in0[inl0], &drv.bias[starting_filter], filter_step * 4);
  inl0 += filter_step;
  memcpy(&in0[inl0], &drv.crf[starting_filter], filter_step * 4);
  inl0 += filter_step;

  memcpy(&in0[inl0], &drv.crx_scale[starting_filter], filter_step * 8);
  inl0 += filter_step * 2;

  memcpy(&in0[inl0], &drv.crx[starting_filter], filter_step * 4);
  inl0 += roundUp(filter_step, 4) / 4;
  drv.mdma->dmas[0].dma_start_send(inl0);
  TOG(cerr << "Starting Send" << endl;);
  drv.mdma->multi_dma_wait_send();
  TOG(cerr << "Finished Send" << endl;);
  // cin.ignore();
  data_transfered += inl0;
  weight_data_sent += inl0;
  wgt_load_calls++;
  prf_end(0, drv.p_t.p_load_wgt);
}

void StartSchedule(acc_container &drv) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int opcode = 16;
  in0[inl0++] = opcode;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->multi_dma_wait_send();
  data_transfered += inl0;
  prf_end(0, drv.p_t.p_start_sched);
}


void LoadInput(acc_container &drv, int starting_row, int number_of_rows,
               int padded_depth) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int inp_packet_a = number_of_rows;
  int inp_packet_b = padded_depth_4 / (UF / 4);
  int inp_packet_c = starting_row;
  int opcode = 4 + 16;
  in0[inl0++] = opcode;
  in0[inl0++] = inp_packet_a;
  in0[inl0++] = inp_packet_b;
  in0[inl0++] = inp_packet_c;
  drv.mdma->dmas[0].dma_change_start(0);
  if (drv.input_preloaded && drv.weight_preloaded) {
    drv.mdma->dmas[0].dma_start_send(inl0);
    drv.mdma->multi_dma_wait_send();
    int src_addr = (starting_row * padded_depth) + DMA_INP_OFFSET;
    drv.mdma->dmas[0].dma_change_start(src_addr);
    drv.mdma->dmas[0].dma_start_send(padded_depth_4 * number_of_rows);
    drv.mdma->multi_dma_wait_send();
  } else {
    int src_addr = (starting_row ) * padded_depth_4;
    memcpy(&in0[inl0], &drv.loaded_inputs[src_addr], padded_depth * number_of_rows);
    inl0 += padded_depth_4 * number_of_rows;
    drv.mdma->dmas[0].dma_start_send(inl0);
    drv.mdma->multi_dma_wait_send();
    data_transfered += inl0;
    input_data_sent += inl0;
    inp_load_calls++;
  }
  drv.mdma->dmas[0].dma_change_start(0);
  prf_end(0, drv.p_t.p_load_inp);
}

void StoreOutTileRow(acc_container &drv, int o_1, int o_3, int filter_step,
                     bool last) {
  prf_start(1);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int schedule = last ? 0 : 16;
  int opcode = 64 + schedule;
  in0[inl0++] = opcode;
  in0[inl0++] = o_1 * drv.ow;
  in0[inl0++] = drv.ow;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->multi_dma_wait_send();
  drv.mdma->dmas[0].dma_start_recv(drv.ow * filter_step + PE_COUNT + 1);
  drv.mdma->multi_dma_wait_recv();
  TOG(cerr << "Getting Output from: " << o_1 * drv.ow;
      cerr << " to " << (o_1 * drv.ow + drv.ow) << endl;);

  int *out0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int8_t *bo0 = reinterpret_cast<int8_t *>(out0);
  int outl0 = 0;
  // for (int o_2 = 0; o_2 < drv.ow; o_2++) {
  //   int o_dex = ((o_1 * drv.ow) + o_2) * drv.oc + o_3;
  //   memcpy(&drv.output_data[o_dex], &out0[outl0], filter_step);
  //   outl0 += filter_step / 4;
  // }
  for (int o_2 = 0; o_2 < drv.ow; o_2++) {
    int o_dex = ((o_1 * drv.ow) + o_2) * drv.oc + o_3;
    for (int f = 0; f < filter_step; f++) {
      drv.output_data[o_dex + f] = out0[outl0++];
      // cout << (int)drv.output_data[o_dex + f] << ",";
    }
    // cout << endl;
  }
  write_data_recv += outl0;
  prf_end(1, drv.p_t.p_store);
}

void MM2IM_Inner_Threaded(acc_container &drv, int o_3, int padded_depth,
                          int filter_step) {
  TOG(cerr << "Starting Threaded Load and Store" << endl;);
  std::vector<Task *> tasks;
  auto *workers_pool = drv.mt_context->workers_pool();
  SRA->o_3 = o_3;
  tasks.push_back(LSA);
  tasks.push_back(SRA);
  workers_pool->Execute(tasks);
  TOG(cerr << "Finished Threaded Load and Store" << endl;);
}

// Current Driver assumes filter_step is divisible by x  (x = 8 currently)
// We need to handle the case where it is not
// By processing the last filter_step % x filters separately
void TileMM2IM(acc_container &drv, int padded_depth) {
  // weight params
  int data_per_filter = drv.ks * drv.ks * padded_depth;
  int cols_per_filter = drv.ks * drv.ks;
  int acc_weight_cols_sup = PE_WGTCOLBUF_SIZE * UF * PE_COUNT;
  int filter_step = min(acc_weight_cols_sup / data_per_filter, PE_COUNT);
  assert(filter_step == PE_COUNT);

  // input params
  int size_per_input_row = padded_depth;
  assert((size_per_input_row / UF) <= PE_INPROWBUF_SIZE);

  // filter params
  int remaining_filters = drv.oc % filter_step;
  int acc_filters = drv.oc - remaining_filters;

  // ==============================================
  int o_3 = 0;

  TOG(cerr << "Starting  MM2IM" << endl;);
  LoadConfig(drv, padded_depth);
  for (; o_3 < drv.oc; o_3 += filter_step) {
    int fs_rem = min(filter_step, drv.oc - o_3);
    // Send filter_step * cols_per_filter  rows of weights to accelerator
    TOG(cerr << "Sending weights: " << o_3 * cols_per_filter << " to "
             << (o_3 * cols_per_filter + fs_rem * cols_per_filter) << endl;);
    LoadWeight(drv, o_3 * cols_per_filter, fs_rem * cols_per_filter,
               padded_depth, fs_rem, o_3);
    TOG(cerr << "Starting Schedule" << endl;);
    StartSchedule(drv);
    int starting = 0;
    for (int o_1 = 0; o_1 < drv.oh; o_1++) {
      TOG(cerr << "Sending rows: " << starting << " to " << drv.oh_ends[o_1] + 1
               << endl;);
      int rows_to_send = drv.oh_ends[o_1] + 1 - starting;
      if (drv.oh_ends[o_1] != starting - 1) {
        LoadInput(drv, starting, rows_to_send, padded_depth);
        TOG(cerr << "Input loaded" << endl;);
      }
      // Last ejects acc from schedule mode
      bool last = (o_1 == drv.oh - 1);
      TOG(cerr << "Pre Store" << endl;);
      StoreOutTileRow(drv, o_1, o_3, fs_rem, last);
      TOG(cerr << "Post Store" << endl;);
      starting = drv.oh_ends[o_1] + 1;
    }
  }
  // SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
};

void Entry(acc_container &drv) {
  int rrows = roundUp(drv.rows, 4);
  int rcols = roundUp(drv.cols, 4);
  int padded_depth = roundUp(drv.depth, UF);
  int output_stride = drv.cols;
  TileMM2IM(drv, padded_depth);
  drv.p_t.inp_data_sent = input_data_sent * 4;
  drv.p_t.wgt_data_sent = weight_data_sent * 4;
  drv.p_t.colmap_data_sent = colmap_data_sent * 4;
  drv.p_t.inp_load_calls = inp_load_calls;
  drv.p_t.wgt_load_calls = wgt_load_calls;
  drv.p_t.colmap_load_calls = colmap_load_calls;
}

} // namespace mm2im_driver

#endif // MM2IM_DRIVER
