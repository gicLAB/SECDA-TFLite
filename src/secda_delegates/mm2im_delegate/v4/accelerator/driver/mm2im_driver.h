#ifndef MM2IM_DRIVER
#define MM2IM_DRIVER

#include "acc_container.h"
#include "mm2im_mt.h"
#include "mm2im_util.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
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
  in0[inl0++] = padded_depth / 16;
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
  prf_end(0, drv.p_t.load_config);
}

void LoadWeight(acc_container &drv, int starting_row, int number_of_rows,
                int padded_depth, int filter_step, int starting_filter) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int opcode = 2;
  int wgt_packet_a = number_of_rows;
  int wgt_packet_b = padded_depth_4 / 4;

  in0[inl0++] = opcode;
  in0[inl0++] = wgt_packet_a;
  in0[inl0++] = wgt_packet_b;
  in0[inl0++] = filter_step;
  for (int i = 0; i < number_of_rows; i++) {
    int src_addr = (starting_row + i) * padded_depth_4;
    memcpy(&in0[inl0], &drv.loaded_weights[src_addr], padded_depth_4 * 4);
    inl0 += padded_depth_4;
    in0[inl0++] = drv.acc_wt_sum[starting_row + i] * drv.rhs_offset;
  }

  // Send Bias
  for (int i = 0; i < filter_step; i++) {
    in0[inl0++] = drv.bias[starting_filter + i];
    in0[inl0++] = drv.crf[starting_filter + i];
    in0[inl0++] = (int)drv.crx[starting_filter + i];
  }
  TOG(cerr << "Sending weights: " << starting_row << " to "
           << (starting_row + number_of_rows) << endl;);
  drv.mdma->dmas[0].dma_start_send(inl0);
  TOG(cerr << "Starting Send" << endl;);
  drv.mdma->multi_dma_wait_send();
  TOG(cerr << "Finished Send" << endl;);
  data_transfered += inl0;
  weight_data_sent += inl0;
  wgt_load_calls++;
  prf_end(0, drv.p_t.load_wgt);
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
  prf_end(0, drv.p_t.start_sched);
}

void LoadInput(acc_container &drv, int starting_row, int number_of_rows,
               int padded_depth) {
  prf_start(0);
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int padded_depth_4 = padded_depth / 4;
  int inp_packet_a = number_of_rows;
  int inp_packet_b = padded_depth_4 / 4;
  int inp_packet_c = starting_row;

  int opcode = 4 + 16;
  in0[inl0++] = opcode;
  in0[inl0++] = inp_packet_a;
  in0[inl0++] = inp_packet_b;
  in0[inl0++] = inp_packet_c;

  for (int i = 0; i < number_of_rows; i++) {
    int src_addr = (starting_row + i) * padded_depth_4;
    memcpy(&in0[inl0], &drv.loaded_inputs[src_addr], padded_depth_4 * 4);
    inl0 += padded_depth_4;
  }
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->multi_dma_wait_send();
  data_transfered += inl0;
  input_data_sent += inl0;
  inp_load_calls++;
  prf_end(0, drv.p_t.load_inp);
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

  int *out0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int8_t *bo0 = reinterpret_cast<int8_t *>(out0);
  int outl0 = 0;
  for (int o_2 = 0; o_2 < drv.ow; o_2++) {
    int o_dex = ((o_1 * drv.ow) + o_2) * drv.oc + o_3;
    memcpy(&drv.output_data[o_dex], &out0[outl0], filter_step);
    outl0 += filter_step / 4;
    // for (int fs = 0; fs < filter_step; fs++) {
    //   int curr = o_dex + fs;
    //   int8_t *out = &drv.output_data[0];
    //   out[curr] = bo0[outl0++];
    //   cout << "output_data[" << curr << "] = " << (int)out[curr] << endl;
    // }
  }
  write_data_recv += outl0;
  prf_end(1, drv.p_t.store);
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

// Current Driver assumes filter_step is divisible by x  (x = 2 currently)
// We need to handle the case where it is not
// By processing the last filter_step % x filters separately
void TileMM2IM(acc_container &drv, int padded_depth) {
  prf_start(1);
  int output_rows = drv.oh * drv.ow;
  int output_cols = drv.oc;

  // weight params
  int data_per_filter = drv.ks * drv.ks * padded_depth;
  int total_weights = data_per_filter * output_cols;
  int rows_per_filter = drv.ks * drv.ks;
  int acc_max_weight_rows = WGT_BUF_LEN * UF / padded_depth;
  int total_weight_rows = total_weights / padded_depth;
  int max_weight_rows = min(total_weight_rows, acc_max_weight_rows);
  int filter_step = min(max_weight_rows / rows_per_filter, PE_COUNT);
  if (rows_per_filter >= max_weight_rows) {
    cerr << "Warning: rows_per_filter: " << rows_per_filter
         << " >= max_weight_rows: " << max_weight_rows << endl;
    filter_step = PE_COUNT;
  }

  // input params
  int max_input_rows_per_output = (drv.ks * drv.ks) / (drv.sx * drv.sy);
  int total_inputs = drv.ih * drv.iw * padded_depth;
  int acc_max_input_rows = INP_BUF_LEN * UF / padded_depth;
  int total_input_rows = total_inputs / padded_depth;
  int max_input_rows = min(total_input_rows, acc_max_input_rows);
  int input_steps = min(max_input_rows, max_input_rows_per_output);

  int padded_out_width = drv.ow + drv.pl + drv.pr;
  int padded_out_height = drv.oh + drv.pt + drv.pb;
  int noOfStepsX = nofSteps(padded_out_width, drv.sx, drv.ks);
  int noOfStepsY = nofSteps(padded_out_height, drv.sy, drv.ks);
  int max_input_rows_per_o1 = noOfStepsX * ceiling(drv.ks, drv.sy);
  if (max_input_rows != total_input_rows &&
      max_input_rows_per_output > max_input_rows)
    cerr << "Warning: max_input_rows_per_output: " << max_input_rows_per_output
         << " > max_input_rows: " << max_input_rows << endl;

  if (max_input_rows != total_input_rows &&
      max_input_rows_per_o1 > max_input_rows)
    cerr << "Warning: max_input_rows_per_o1: " << max_input_rows_per_o1
         << " > max_input_rows: " << max_input_rows << endl;
  int input_o1_steps = min(max_input_rows, max_input_rows_per_o1);

  // filter params
  int remaining_filters = output_cols % filter_step;
  int acc_filters = output_cols - remaining_filters;

  // ==============================================
  // drv.validate();
  int o_3 = 0;
  if (drv.thread_count > 1) {
    LSA = new struct Load_Send_Acc(drv, padded_depth);
    SRA = new struct Store_Results_Acc(drv, o_3, filter_step);
  }
  if (output_cols >= PE_COUNT) {
    TOG(cerr << "Starting  MM2IM" << endl;);
    if (acc_filters > 0) LoadConfig(drv, padded_depth);

    for (; o_3 < acc_filters; o_3 += filter_step) {
      // Send filter_step * rows_per_filter  rows of weights to accelerator
      TOG(cerr << "Sending weights: " << o_3 * rows_per_filter << " to "
               << (o_3 * rows_per_filter + filter_step * rows_per_filter)
               << endl;);
      LoadWeight(drv, o_3 * rows_per_filter, filter_step * rows_per_filter,
                 padded_depth, filter_step, o_3);
      TOG(cerr << "Starting Schedule" << endl;);
      StartSchedule(drv);
      int starting = 0;

      // Dual-Threaded
      if (drv.thread_count > 1)
        MM2IM_Inner_Threaded(drv, o_3, padded_depth, filter_step);
      else {
        for (int o_1 = 0; o_1 < drv.oh; o_1++) {
          TOG(cerr << "Sending rows: " << starting << " to "
                   << drv.oh_ends[o_1] + 1 << endl;);
          int rows_to_send = drv.oh_ends[o_1] + 1 - starting;
          if (drv.oh_ends[o_1] != starting - 1) {
            LoadInput(drv, starting, rows_to_send, padded_depth);
            TOG(cerr << "Input loaded" << endl;);
          }
          // Last ejects acc from schedule mode
          bool last = (o_1 == drv.oh - 1);
          TOG(cerr << "Pre Store" << endl;);
          StoreOutTileRow(drv, o_1, o_3, filter_step, last);
          TOG(cerr << "Post Store" << endl;);
          starting = drv.oh_ends[o_1] + 1;
        }
      }
    }
  }

  // Handle remaining filters
  // vector<vector<int>> &mm2im_map = *drv.mm2im_map;
  // for (; o_3 < output_cols; o_3++) {
  //   for (int o_1 = 0; o_1 < drv.oh; o_1++) {
  //     for (int o_2 = 0; o_2 < drv.ow; o_2++) {
  //       int o_dex = ((o_1 * drv.ow) + o_2) * output_cols + o_3;
  //       int32_t sum = 0;
  //       for (int i = 0; i < mm2im_map[o_dex].size(); i++) {
  //         int orow = mm2im_map[o_dex][i] % drv.rows;
  //         int ocol = mm2im_map[o_dex][i] / drv.rows;
  //         for (int d = 0; d < drv.depth; d++) {
  //           int weight_index = orow * drv.depth + d;
  //           int input_index = ocol * drv.depth + d;
  //           int weight = drv.weights[weight_index];
  //           int input = drv.inputs[input_index];
  //           sum += weight * input;
  //         }
  //         int offset = drv.acc_wt_sum[orow] * drv.rhs_offset;
  //         sum += offset;
  //       }
  //       int bias = drv.bias[o_3];
  //       int crf_data = drv.crf[o_3];
  //       int crx_data = drv.crx[o_3];
  //       int qm_ret =
  //           drv.ra + CPU_Quantised_Multiplier(sum + bias, crf_data, crx_data);
  //       if (qm_ret > MAX8) qm_ret = MAX8;
  //       else if (qm_ret < MIN8) qm_ret = MIN8;
  //       drv.output_data[o_dex] = qm_ret;
  //     }
  //   }
  // }
  SYSC_ON(drv.profile->saveProfile(drv.acc->profiling_vars));
  prf_end(1, drv.p_t.driver_total);
};

void Entry(acc_container &drv) {
  int rrows = roundUp(drv.rows, 4);
  int rcols = roundUp(drv.cols, 4);
  int padded_depth = roundUp(drv.depth, 16);
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