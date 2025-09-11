#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#ifdef SYSC
#include "../acc.sc.h"
#include "systemc_binding.h"
#else
#endif

#include "../acc_config.sc.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/secda_profiler/profiler.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/mman.h>
#include <typeinfo>
#include <unistd.h>
#include <vector>

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;
#define TSCALE microseconds

struct vm_times {
  duration_ns load_send_inputs;
  duration_ns load_send_weights;
  duration_ns send_weights;
  duration_ns set_results;
  duration_ns start_compute;
  duration_ns receive_results;
  duration_ns vm_acc;
  duration_ns store;
  duration_ns ipack;
  duration_ns conv_total;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    prf_out(TSCALE, load_send_inputs);
    prf_out(TSCALE, load_send_weights);
    prf_out(TSCALE, send_weights);
    prf_out(TSCALE, set_results);
    prf_out(TSCALE, start_compute);
    prf_out(TSCALE, receive_results);
    prf_out(TSCALE, store);
    prf_out(TSCALE, vm_acc);
    prf_out(TSCALE, ipack);
    prf_out(TSCALE, conv_total);
    cout << "================================================" << endl;
#endif
  }

  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out(TSCALE, load_send_inputs, file);
    prf_file_out(TSCALE, load_send_weights, file);
    prf_file_out(TSCALE, send_weights, file);
    prf_file_out(TSCALE, set_results, file);
    prf_file_out(TSCALE, start_compute, file);
    prf_file_out(TSCALE, receive_results, file);
    prf_file_out(TSCALE, store, file);
    prf_file_out(TSCALE, vm_acc, file);
    prf_file_out(TSCALE, ipack, file);
    prf_file_out(TSCALE, conv_total, file);
    file.close();
#endif
  }
};

// Used for profiling
struct layer_details {
  int layer = 0;
  int conv_layer_no = 0; // for conv layers
  int node = 0;
  int layer_weight_tile = 0;
  int layer_input_tile = 0;
  unsigned int wgt_tile_offset = 0;           // offset for the weight tile
  unsigned int layer_wgt_dma_curr_offset = 0; // each buffer layer offset
  unsigned int layer_wgt_offsets[500]; // assuming we will not need to allocate
                                       // more than 500
  // layers
  bool layer_wgt_preLoadedToDMA[500]; // to check if the layer is allocated or
                                      // not
  bool profile = false;

  bool alloc_layer(int layer, unsigned int layer_wgt_dma_curr_offset,
                   unsigned int wgt_size) {
    if (((layer_wgt_dma_curr_offset * NO_OF_DATA_CHANNELS * sizeof(int32_t)) +
         wgt_size) >= (DMA_WGT_SIZE_4 * NO_OF_DATA_CHANNELS)) {
      return false;
    }
    layer_wgt_offsets[layer] = layer_wgt_dma_curr_offset;
    layer_wgt_preLoadedToDMA[layer] = true;
    return true;
  }
};

// Used for tracking output locations
struct store_params {
  int *dst;
  int dcs;
  int rows;
  int cols;
  int rcols;
  int rrows;
};

struct acc_container {
#ifdef SYSC
  // Gives SystemC accelerator access
  ACCNAME *acc;
#else
  // Gives accelerator access
  int *acc;
#endif

  Profile *profile;
  // DMAs Pointer
  struct multi_dma *mdma;
  // Accelerator Layer Details
  int op_type;

  // Temporary Weight non-MMapped Padded Buffers
  int *wb_0;
  int *wb_1;
  int *wb_2;
  int *wb_3;

  // Temporary Input non-MMapped Padded Buffers
  int *inb_0;
  int *inb_1;
  int *inb_2;
  int *inb_3;
  int in_id = 0;

  // Driver variables
  struct store_params *st_params;
  MultiThreadContext *mt_context;
  int thread_count;
  int w_c = 0;

  // Output Pipeline Metadata
  vector<int> wt_sum1;
  vector<int> wt_sum2;
  vector<int> wt_sum3;
  vector<int> wt_sum4;
  int *in_sum1;
  int *in_sum2;
  int *in_sum3;
  int *in_sum4;
  int *bias;
  vector<int> crf;
  vector<int8_t> crx;
  int ra;
  int inp_offset = 0;
  int wgt_offset = 0;

  int rows = 0;
  int cols = 0;
  int depth = 0;
  int8_t *dst;

  // Pipeline vars
  struct dma_buffer_set *dfs;
  struct DSR dsr;
  bool wgt_start = false;
  int recv_len;

  // GEMM Info variable
  struct layer_details *t;
  struct vm_times t2;
  bool use_sim = false;

  bool Check_Done() { return (mdma->multi_dma_check_recv() == 0); }

  void End_Transfer() { mdma->multi_dma_wait_send(); }

  bool Start_Transfer() {
    if (!(dsr.sID == dsr.cID && dsr.dID > dsr.sID)) return false;
    int s_buf = find_dbuf(dfs[0], dsr.sID);
    mdma->multi_dma_change_start_4(dfs[0].dbuf_set[s_buf].offset);
    mdma->dmas[0].dma_start_send(dfs[0].dbuf_set[s_buf].len);
    mdma->dmas[1].dma_start_send(dfs[1].dbuf_set[s_buf].len);
    mdma->dmas[2].dma_start_send(dfs[2].dbuf_set[s_buf].len);
    mdma->dmas[3].dma_start_send(dfs[3].dbuf_set[s_buf].len);
    End_Transfer();
    dsr.sID++;
    return true;
  }

  void Set_Results() {
    // int s_buf = find_dbuf(dfs[0], dsr.cID);
    // mdma->multi_dma_change_end(dfs[0].dbuf_set[s_buf].offset);
    mdma->multi_dma_change_end(0);
    mdma->multi_dma_start_recv(recv_len);
    // dsr.cID++;
  }

  void Recieve_Results() { mdma->multi_dma_wait_recv_4(); }
};

//========================//========================//========================//

void pre_load_wgt_toDMA(vector<int8_t> &wb0, vector<int8_t> &wb1,
                        vector<int8_t> &wb2, vector<int8_t> &wb3, int *dims,
                        layer_details *t, multi_dma *mdma) {

  // check NO_OF_DATA_CHANNELS is 4
  assert(NO_OF_DATA_CHANNELS == 4 &&
         "Error: pre_load_wgt_toDMA():NO_OF_DATA_CHANNELS must be 4 for this "
         "function");
  // round up width and depth
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];
  int rwidth = roundUp(width, WGTBLOCK_WIDTH);
  int rdepth = roundUp(depth, BLOCK_DEPTH);

  // Calculate the data size needs to copy this time to DMA
  int dataSize = rwidth * rdepth;
  int dataSizeEachBuff = dataSize / NO_OF_DATA_CHANNELS;
  // DLOG("pre_load_wgt_toDMA: width: " << width << " depth: " << depth
  //                                    << " rwidth: " << rwidth << " rdepth: "
  //                                    << rdepth << " dataSize: " << dataSize);
  // check allocation will be okay or not
  bool check =
      t->alloc_layer(t->conv_layer_no, t->layer_wgt_dma_curr_offset, dataSize);
  assert(check && "Error: pre_load_wgt_toDMA(): "
                  "alloc_layer failed");

  // DLOG("conv_layer_no: " << t->conv_layer_no << " layer_wgt_offsets: "
  //                        << t->layer_wgt_offsets[t->conv_layer_no]);

  //
  int *in0 = mdma->dmas[0].dma_get_inbuffer() + DMA_SCRATCH_SIZE_4 +
             t->layer_wgt_dma_curr_offset;
  int *in1 = mdma->dmas[1].dma_get_inbuffer() + DMA_SCRATCH_SIZE_4 +
             t->layer_wgt_dma_curr_offset;
  int *in2 = mdma->dmas[2].dma_get_inbuffer() + DMA_SCRATCH_SIZE_4 +
             t->layer_wgt_dma_curr_offset;
  int *in3 = mdma->dmas[3].dma_get_inbuffer() + DMA_SCRATCH_SIZE_4 +
             t->layer_wgt_dma_curr_offset;
  // reinterpret cast to int8_t pointer
  int8_t *in0_ptr = reinterpret_cast<int8_t *>(in0);
  int8_t *in1_ptr = reinterpret_cast<int8_t *>(in1);
  int8_t *in2_ptr = reinterpret_cast<int8_t *>(in2);
  int8_t *in3_ptr = reinterpret_cast<int8_t *>(in3);

  // use memcpy to copy the data from wb0, wb1, wb2, wb3
  // to in0_ptr, in1_ptr, in2_ptr, in3_ptr because in
  // here wb0, wb1, wb2, wb3 are pointing to the 0
  // address
  memcpy(in0_ptr, wb0.data(), dataSizeEachBuff);
  memcpy(in1_ptr, wb1.data(), dataSizeEachBuff);
  memcpy(in2_ptr, wb2.data(), dataSizeEachBuff);
  memcpy(in3_ptr, wb3.data(), dataSizeEachBuff);

  // update prev_offset
  // dataSizeEachBuff is in no. of bytes, since we are
  // using int8 weights no. of elements are same as no.
  // of bytes we are pointing in0, in1, in2, in3 using
  // int pointers
  t->layer_wgt_dma_curr_offset += dataSizeEachBuff / sizeof(int32_t);
}

#ifndef potTest

int8_t roundToNearestPOTLevels(double value, const int8_t *pot_levels,
                               int size) {
  int8_t nearest = pot_levels[0];
  double min_diff = abs(value - nearest);
  for (int i = 1; i < size; ++i) {
    double diff = abs(value - pot_levels[i]);
    if (diff < min_diff) {
      min_diff = diff;
      nearest = pot_levels[i];
    }
  }
  return nearest;
}

bool check_pot_weights(const int8_t *wgt_pot, int *dims,
                       const int8_t *pot_levels, int pot_levels_size) {
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];
  bool is_this_potChannel = false;
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < depth; ++j) {
      int index = i * depth + j;
      is_this_potChannel = false;
      for (int k = 0; k < pot_levels_size; ++k) {
        if (pot_levels[k] == wgt_pot[index]) {
          is_this_potChannel = true;
          break;
        }
      }
      if (!is_this_potChannel) {
        // cout << "filter[" << i << "] is this_potChannel: " <<
        // is_this_potChannel
        //      << endl;
        // cout << "filter[" << i << "][" << j << "] = " << (int)wgt_pot[index]
        //      << " is not in pot_levels: ";
        // for (int k = 0; k < pot_levels_size; ++k) {
        //   cout << (int)pot_levels[k] << ",";
        // }
        // cout << endl;
        return false;
      }
    }
    // cout << "filter[" << i << "] is this_potChannel: " << is_this_potChannel
    //      << endl;
  }
  return true;
}

double find_golden_scale(TfLiteTensor *filter, const int8_t *pot_levels,
                         int pot_levels_size) {
  double golden_scale = 0.0;
  int *dims = filter->dims->data;
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];
  auto *affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
  float *filter_scales = affine_quantization->scale->data;
  int8_t *filter_data = filter->data.int8;
  bool is_this_goldenChannel = false;
  for (int i = 0; i < width; ++i) {
    golden_scale = static_cast<double>(filter_scales[i]);
    for (int j = 0; j < depth; ++j) {
      int index = i * depth + j;
      // in the depth dimension, check all the int8_t values are within the apot
      // levels
      is_this_goldenChannel = false;
      for (int k = 0; k < pot_levels_size; ++k) {
        if (pot_levels[k] == filter_data[index]) {
          is_this_goldenChannel = true;
          break;
        }
      }
      if (!is_this_goldenChannel) {
        continue; // skip this channel if not golden
      }
    }
    cout << "is_this_goldenChannel: " << is_this_goldenChannel << endl;
    // if (is_this_goldenChannel) {
    //   return golden_scale;
    // }
  }
  return golden_scale; // return the last golden scale found
  assert(is_this_goldenChannel &&
         "Error: find_golden_scale(): No golden channel found in the filter.");
}

double find_scale_n_pot_weights(int8_t *wgt_arr, int *dims, double filter_scale,
                                const int8_t *pot_levels, int pot_levels_size,
                                int8_t *pot_wgts = nullptr) {
  // Find the scale for which int8_levels will have minimum mismatch
  // with the pot_levels
  double pot_scale = 0.0;

  // find minimum and maximum values in wgt_pot
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];

  // we assume pot_levels are symmetric around 0
  int pot_levels_length = pot_levels_size / 2;
  // current scale is for pot_levels[0]
  // therefore, starting from 1
  int min_mismatches = 16; // maximum no. of mismatches we can have
  double pot_scale_for_min_mismatches = 0.0;
  int8_t minDiff = 0;
  int8_t maxDiff = 0;
  // create a vector to store the int8_t levels
  vector<int8_t> int8_levels(width * depth);
  for (int i = 1; i < pot_levels_length; ++i) {
    pot_scale = (filter_scale * 254.0) /
                (static_cast<double>(pot_levels[i]) * (-1.0) * 2.0);
    // cout << "pot_scale: " << pot_scale << endl;
    // convert pot_scale to int8_t levels

    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < depth; ++k) {
        int index = j * depth + k;
        // dequantize the data
        double dequantized_value = static_cast<double>(wgt_arr[index]) *
                                   static_cast<double>(filter_scale);
        // quantize the data using pot_scale
        // Quantize and clamp to int8_t range
        // round to nearest integer
        int quantized = static_cast<int>(round(dequantized_value / pot_scale));
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        int8_t quantized_value = static_cast<int8_t>(quantized);
        int8_levels[index] = quantized_value;
      }
    }
    // find unique values in int8_levels
    std::set<int8_t> unique_int8_levels(int8_levels.begin(), int8_levels.end());

    // std::cout << "Unique levels in int8_levels:" << std::endl;
    // for (int8_t val : unique_int8_levels) {
    //   std::cout << static_cast<int>(val)
    //             << " "; // cast to int to print properly
    // }
    // std::cout << std::endl;
    std::vector<int8_t> unique_int8_levels_vec(unique_int8_levels.begin(),
                                               unique_int8_levels.end());
    // return the size of unique_int8_levels
    int unique_size = static_cast<int>(unique_int8_levels_vec.size());
    // cout << "Unique int8 levels size: " << unique_size << endl;
    // check if the unique_int8_levels are within the pot_levels
    // create a dim to pass
    int *dims_temp = new int[4];
    dims_temp[0] = 1;           // width
    dims_temp[1] = unique_size; // depth
    dims_temp[2] = 1;
    dims_temp[3] = 1;

    bool is_this_potChannel = check_pot_weights(
        unique_int8_levels_vec.data(), dims_temp, pot_levels, pot_levels_size);
    if (is_this_potChannel) {
      // cout << "Found a Scale for Pot Weights: " << pot_scale << endl;
      // if yes, return the pot_scale

      // copy the int8_levels to pot_wgts
      for (int j = 0; j < width * depth; ++j) {
        // pot_wgts[j] = int8_levels[j];
        // round to nearest pot_levels
        pot_wgts[j] = roundToNearestPOTLevels(
            static_cast<double>(int8_levels[j]), pot_levels, pot_levels_size);
      }
      return pot_scale;
    } else {
      // algorithm:
      // 1. print unique values in int8_levels
      // for (int8_t val : unique_int8_levels_vec) {
      //   cout << static_cast<int>(val) << " ";
      // }
      // cout << endl;
      // 2. check how many unique values are not in pot_levels for each
      // pot_scale
      std::vector<int8_t> mismatches;
      std::vector<int8_t> closest_pot_levels;
      for (int8_t val : unique_int8_levels_vec) {
        if (std::find(pot_levels, pot_levels + pot_levels_size, val) ==
            pot_levels + pot_levels_size) {
          mismatches.push_back(val);
          // find the closest value in pot_levels for this mismatch
          int8_t closest = pot_levels[0];
          double min_diff = abs(val - closest);
          for (int k = 1; k < pot_levels_size; ++k) {
            double diff = abs(val - pot_levels[k]);
            if (diff < min_diff) {
              min_diff = diff;
              closest = pot_levels[k];
            }
          }
          closest_pot_levels.push_back(closest);
        }
      }
      // 3. take the lowest no. of mismatches from all the pot_scales
      // 4. return the pot_scale with lowest no. of mismatches
      if (mismatches.size() < min_mismatches) {
        min_mismatches = mismatches.size();
        pot_scale_for_min_mismatches = pot_scale;

        std::vector<int8_t> diff;
        for (size_t j = 0; j < mismatches.size(); ++j) {
          // calculate the difference between mismatches and
          // closest_pot_levels
          int8_t diff_value = mismatches[j] - closest_pot_levels[j];
          diff.push_back(diff_value);
        }
        // find the minDiff and maxDiff
        minDiff = *std::min_element(diff.begin(), diff.end());
        maxDiff = *std::max_element(diff.begin(), diff.end());
        // copy the int8_levels to pot_wgts
        for (int k = 0; k < width * depth; ++k) {
          // pot_wgts[k] = int8_levels[k];
          // round to nearest pot_levels
          pot_wgts[k] = roundToNearestPOTLevels(
              static_cast<double>(int8_levels[k]), pot_levels, pot_levels_size);
        }
        // if (maxDiff > 1) {
        //   // cout bot mismatch values and their closest pot_levels
        //   cout << "Mismatch values and their closest pot_levels: " << endl;
        //   for (size_t j = 0; j < mismatches.size(); ++j) {
        //     cout << "Mismatch: " << (int)mismatches[j]
        //          << " Closest Pot Level: " << (int)closest_pot_levels[j]
        //          << endl;
        //   }
        // }
      }
    }
  }
  // cout << endl;
  cout << "Found a Scale for Pot Weights with minimum mismatches: "
       << pot_scale_for_min_mismatches << " with mismatches: " << min_mismatches
       << " minDiff: " << (int)minDiff << " maxDiff: " << (int)maxDiff << endl;

  // if no scale found, return the last pot_scale
  return pot_scale_for_min_mismatches;
}

void prepare_pot_weights_v2(TfLiteTensor *filter, int8_t *wgt_pot = nullptr,
                            const int8_t *pot_levels = nullptr,
                            int pot_levels_size = 15) {
  int *dims = filter->dims->data;
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];
  int8_t *filter_data = filter->data.int8;
  auto *affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
  float *filter_scales = affine_quantization->scale->data;

  for (int i = 0; i < width; ++i) {
    // check all the int8_t values in the depth dimension are within the
    // pot_levels
    bool is_this_potChannel = false;
    // create a dim to pass to the check_pot_weights function
    int dims_temp[4] = {1, dims[1], dims[2], dims[3]};
    is_this_potChannel = check_pot_weights(filter_data + i * depth, dims_temp,
                                           pot_levels, pot_levels_size);
    // cout << "filter[" << i << "] is this_potChannel: " << is_this_potChannel
    //      << endl;
    ////////////////////////////////////////////////////
    // cout << "Channel " << i << " scale: " << filter_scales[i] << " || ";
    // for (int j = 0; j < depth; ++j) {
    //   cout << (int)filter_data[i * depth + j] << ",";
    //   if (j >= 5) break; // limit output to first 10 values
    // }
    // cout << " || ";
    ////////////////////////////////////////////////
    if (is_this_potChannel) {
      // if yes, no need to change the scale, only copy the int8 levels
      // to the wgt_pot vector
      for (int j = 0; j < depth; ++j) {
        wgt_pot[i * depth + j] = filter_data[i * depth + j];
      }
    } else {
      // if no, find the scale for which int8_levels will have minimum
      // mismatch with the apot_levels
      double pot_scale = 0.0;
      pot_scale = find_scale_n_pot_weights(
          filter_data + i * depth, dims_temp,
          static_cast<double>(filter_scales[i]), pot_levels, pot_levels_size,
          wgt_pot ? wgt_pot + i * depth : nullptr);
      // cout << "filter[" << i << "] scale: " << filter_scales[i]
      //      << " pot_scale: " << pot_scale << endl;
      filter_scales[i] = pot_scale;
    }

    ////////////////////////////////////////////////////
    // cout << "New scale: " << filter_scales[i] << " || ";
    // for (int j = 0; j < depth; ++j) {
    //   cout << (int)wgt_pot[i * depth + j] << ",";
    //   if (j >= 16) break; // limit output to first 10 values
    // }
    // cout << endl;
    ////////////////////////////////////////////////////
  }
}

void prepare_pot_weights(TfLiteTensor *filter, int8_t *wgt_pot = nullptr,
                         double golden_scale = 0.0,
                         const int8_t *pot_levels = nullptr,
                         int pot_levels_size = 15) {
  int *dims = filter->dims->data;
  int width = dims[0];
  int depth = dims[1] * dims[2] * dims[3];
  int8_t *filter_data = filter->data.int8;
  auto *affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
  float *filter_scales = affine_quantization->scale->data;

  for (int i = 0; i < width; ++i) {
    cout << "Channel " << i << " scale: " << filter_scales[i] << " || ";
    for (int j = 0; j < depth; ++j) {
      cout << (int)filter_data[i * depth + j] << ",";
      if (j >= 5) break; // limit output to first 10 values
    }
    cout << " || ";
    // cout << " value not in approxPattern: ";
    cout << "New scale: " << golden_scale << " || ";
    for (int j = 0; j < depth; ++j) {
      // dequantize the data
      double dequantized_value =
          static_cast<double>(filter_data[i * depth + j]) *
          static_cast<double>(filter_scales[i]);
      // quantize the data using golden scale
      // int8_t quantized_value =
      //     static_cast<int8_t>(round(dequantized_value / golden_scale));
      int8_t quantized_value = roundToNearestPOTLevels(
          dequantized_value / golden_scale, pot_levels, pot_levels_size);
      // fill the wgt_pot with the quantized value
      wgt_pot[i * depth + j] = quantized_value;
      // wgt_pot[i * depth + j] = filter_data[i * depth + j]; // for testing
    }
    for (int j = 0; j < depth; ++j) {
      cout << (int)wgt_pot[i * depth + j] << ",";
      if (j >= 16) break; // limit output to first 10 values
    }
    cout << endl;
    // fill the scale value with golden scale
    filter_scales[i] = golden_scale;
  }
}

#endif // ! potTest

void preload_weights(int8_t *weight_data, int *dims, vector<int8_t> &wb0,
                     vector<int8_t> &wb1, vector<int8_t> &wb2,
                     vector<int8_t> &wb3, vector<int> &wt_sum1,
                     vector<int> &wt_sum2, vector<int> &wt_sum3,
                     vector<int> &wt_sum4, int inpZeroPoint, int *bias) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        int8_t weights[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
      } else {
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
      }
    }

    wt_sum1.push_back((s0 * inpZeroPoint) + bias[(i * 4) + 0]);
    wt_sum2.push_back((s1 * inpZeroPoint) + bias[(i * 4) + 1]);
    wt_sum3.push_back((s2 * inpZeroPoint) + bias[(i * 4) + 2]);
    wt_sum4.push_back((s3 * inpZeroPoint) + bias[(i * 4) + 3]);
  }
}

void precal_sum_load_pad(const int8_t *data, int width, int depth, int8_t *inb0,
                         int8_t *inb1, int8_t *inb2, int8_t *inb3) {
  int w = ((width + 3) - ((width + 3) % 4));
  int d = ((depth + 15) - ((depth + 15) % 16));
  int d2 = depth * 2;
  int d3 = depth * 3;
  int d4 = depth * 4;
  int i_c = 0;
  int sums_curr = 0;

  const int8_t *inp_d = reinterpret_cast<const int8_t *>(data);
  int dm = 0;
  for (int i = 0; i < w / 4; i++) {
    int id = i * d4;
    int i0 = id;
    int i1 = id + depth;
    int i2 = id + d2;
    int i3 = id + d3;
    int ss0 = 0;
    int ss1 = 0;
    int ss2 = 0;
    int ss3 = 0;

#ifdef ACC_NEON
    dm = d - 16;
    int8x16_t tmp0;
    int8x16_t tmp1;
    int8x16_t tmp2;
    int8x16_t tmp3;

    for (int j = 0; j < dm; j += 16) {
      tmp0 = vld1q_s8(inp_d + i0 + j);
      tmp1 = vld1q_s8(inp_d + i1 + j);
      tmp2 = vld1q_s8(inp_d + i2 + j);
      tmp3 = vld1q_s8(inp_d + i3 + j);
      vst1q_s8(inb0 + i_c, tmp0);
      vst1q_s8(inb1 + i_c, tmp1);
      vst1q_s8(inb2 + i_c, tmp2);
      vst1q_s8(inb3 + i_c, tmp3);
      i_c += 16;
    }

#endif
    for (int j = dm; j < d; j++) {
      if (j < depth) {
        unsigned char w0 = data[i0 + j];
        unsigned char w1 = data[i1 + j];
        unsigned char w2 = data[i2 + j];
        unsigned char w3 = data[i3 + j];

        inb0[i_c] = w0;
        inb1[i_c] = w1;
        inb2[i_c] = w2;
        inb3[i_c++] = w3;
      } else {
        inb0[i_c] = 0;
        inb1[i_c] = 0;
        inb2[i_c] = 0;
        inb3[i_c++] = 0;
      }
    }
  }
}

#endif // ACC_CONTAINER