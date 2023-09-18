// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * This file defines PaddlePredictor, the api for lite. It supports multiple
 * hardware including ARM, X86, OpenCL, CUDA and so on.
 */

#ifndef PADDLE_LITE_API_2_H_  // NOLINT
#define PADDLE_LITE_API_2_H_
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle_place.h"  // NOLINT
#include "paddle_api.h"


namespace paddle {
namespace lite_api {

#ifdef LITE_WITH_ARM

LITE_API     void paddle_DeviceInit();
LITE_API     void paddle_clip_kernel_fp32(const float* input, int64_t num, float min, float max, float* output);
LITE_API     void paddle_elementwise_mul(const float* dinx, const float* diny, float* dout, int num);
LITE_API     void paddle_elementwise_div(const float* dinx, const float* diny, float* dout, int num);
LITE_API     void paddle_elementwise_add(const float* dinx, const float* diny, float* dout, int num);
LITE_API     void paddle_elementwise_sub(const float* dinx, const float* diny, float* dout, int num);
LITE_API     void paddle_elementwise_pow(const float* dinx, const float* diny, float* dout, int num);
LITE_API     void paddle_elementwise_max(const float* dinx, const float* diny, float* dout,int num);
LITE_API     void paddle_act_relu(const float* din, float* dout, int size, int threads);
LITE_API     void paddle_act_sigmoid(const float* din, float* dout, int size, int threads);
LITE_API     void act_tanh(const float* din, float* dout, int size, int threads);
LITE_API     void act_log(const float* din, float* dout, int size, int threads);
LITE_API     void act_exp(const float* din, float* dout, int size, int threads);

// New added activate
LITE_API     void act_leakyrelu(const float* din, float* dout, int size, float alpha, int threads);
LITE_API     void act_sqrt(const float* din, float* dout, int size, int threads);
LITE_API     void act_softmax(const float* din, float* dout, int dims, int axis_num);


LITE_API     void scale(const float* din, float* dout, int num, float scale, float bias);
LITE_API     void scale(const int* din, int* dout, int num, int scale, int bias);
LITE_API     void scale(const float* din,
                  float* dout,
                  int outer_dim,
                  int scale_dim,
                  int inner_dim,
                  const float* scale_data,
                  const float* bias_data);
LITE_API     void scale(const float* din,
                  float* dout,
                  int outer_dim,
                  int scale_dim,
                  const float* scale_data,
                  const float* bias_data);

LITE_API void paddle_matmul(const int M, const int N, const int K, const float* X, const float* W, float* Y, int cls=0, int ths=1);
LITE_API void paddle_fccompute(const int M, const int N, const int K,
      const float* X, const float* W, float* Y,
      const float* bias = nullptr, ActivationType activationtype = ActivationType::kIndentity,
      int cls=0, int ths=1);

LITE_API void paddle_matmul_quantize(const int M, const int N, const int K,
        const int8_t* X, const float xscale,
        const int8_t* W, const float wscale, float* Y);

LITE_API  void paddle_conv1d(
      int batches,
      int channels, int xlen, float* indata,
      int filters, int kernelsize, float *kerneldata,
      float*outdata, float* bias=NULL,
      int padding = 0, int dilation = 1, int stride = 1,
      int flag_act = 0, float leaky_relu_scale = 0.1,
      int cls = 1, int ths = 2);

LITE_API void paddle_conv2d(
      int batches,
      int x_h, int x_w, int channels, float* indata,
      int filters, int kernel_h, int kernel_w, float* kerneldata,
      float* outdata, float* bias=NULL,
      int padding_h = 0, int padding_w = 0, 
      int dilation_h = 1, int dilation_w = 1,
      int stride_h = 1, int stride_w = 1,
      int flag_act = 0, float leaky_relu_scale = 0.1,
      int cls = 1, int ths = 2);

LITE_API  void paddle_conv(
      std::vector<int64_t>indatashape, float* indata,
      std::vector<int64_t>kernelshape, float *kerneldata,
      std::vector<int64_t>outdatashape, float*outdata,
      bool flag_bias, float*biasdata,
      std::vector<int> pad,
      std::vector<int> dilation,
      std::vector<int> stride, 
      int flag_act, float leaky_relu_scale, int cls, int ths);
LITE_API void paddle_matrix_norm_row(const float* x_data,
                     const float* scale_data,
                     const float* bias_data,
                     float* out_data,
                     float* mean_out,
                     float* var_out,
                     float epsilon,
                     int batch_size,
                     int feature_size);
LITE_API void paddle_mean_var(const float* x_data,
                     float* mean_out,
                     float* var_out,
                     float epsilon,
                     int batch_size,
                     int feature_size);
// LITE_API void paddle_conv1d_int(
//       int channels, int xlen, int8_t* indata,
//       int filters, int kernelsize, int8_t *kerneldata,
//       float *outdata, float* bias,  float input_scale, float weight_scale, 
//       int padding=0, int dilation=1, int stride=1,
//       int flag_act=0, float leaky_relu_scale=0.1,
//       int cls=0, int ths=1);

// LITE_API void paddle_conv_int(
//       std::vector<int64_t>indatashape, int8_t* indata,
//       std::vector<int64_t>kernelshape, int8_t *kerneldata,
//       std::vector<int64_t>outdatashape, float*outdata,
//       bool flag_bias, float*biasdata,float input_scale,  float weight_scale, 
//       std::vector<int> pad,
//       std::vector<int> dilation,
//       std::vector<int> stride, int flag_act, float leaky_relu_scale, int cls=0, int ths=1);











LITE_API  void paddle_conv_transpose1d(
      int channels, int xlen, float* indata,
      int filters, int kernelsize, float* kerneldata,
      float* outdata,
      int padding = 0, int dilation = 1, int stride = 1,
      int flag_act = 0, float leaky_relu_scale = 0.1,
      int cls = 1, int ths = 2
     );


LITE_API void paddle_conv_transpose2d(
      int channels, int x_h, int x_w, float* indata,
      int filters, int kernel_h, int kernel_w, float* kerneldata,
      float* outdata,
      int padding_h=0, int padding_w=0, 
      int dilation_h=1, int dilation_w=1,
      int stride_h=1, int stride_w=1,
      int flag_act=0, float leaky_relu_scale=0.1,
      int cls=1, int ths=2);
LITE_API void paddle_conv_transpose(
          std::vector<int64_t>indatashape, float* indata,
          std::vector<int64_t>kernelshape, float* kerneldata,
          std::vector<int64_t>outdatashape, float* outdata,
          std::vector<int> pad,
          std::vector<int> dilation,
          std::vector<int> stride, int flag_act, float leaky_relu_scale, int cls, int ths);

LITE_API void paddle_layernorm1d(float* x, float* weight, float* bias, float* outdata, 
      float* meandata, float* vardata, int batch_size, int features);

LITE_API void paddle_batchnorm1d(float* x, float* outdata, 
      float* scale, float* bias, float* mean_data, float* var_data,
      int channels, int xlen, int cls = 1, int ths = 2);

LITE_API void paddle_batchnorm(std::vector<int64_t>indatashape, float* indata, float* outdata,
          float* scaledata, float*biasdata, float* mean_data, float* var_data,
          int cls = 1, int ths = 2);

LITE_API void paddle_fill_bias(float* x, float* bias, int channels, int xlen, bool flag_relu=false);

LITE_API void paddle_transpose2d(float* x, float* out, int size1, int size2, int cls = 0, int ths = 1);

LITE_API void paddle_transpose3d(float* x, float* out, int size1, int size2, int size3, int axis1, int axis2, int cls = 0, int ths = 1);

LITE_API void paddle_transpose(float* x, float* out, std::vector<int> axis_size, int axis1 = 1, int axis2 = 0, int cls = 0, int ths=1);

LITE_API void paddle_transpose(std::vector<int64_t> input_shape, float* indata, 
      std::vector<int64_t> output_shape, float* outdata,
      std::vector<int> axis, int cls, int ths);

LITE_API void paddle_reflect1d(float* din, float* dout, int channels, int x_len, int dilation);
LITE_API void paddle_reflect2d(float* din, float* dout, int channels, int x_h, int x_w, int dilation_h, int dilation_w);

LITE_API void paddle_matmul_int16_32(int m, int n, int k, int16_t* A, int16_t* B, int32_t* C, bool rettrans, int cls=1, int ths=2);

// New added functions
LITE_API void paddle_affine(const float* din, const float* weight, const float* bias, const int dim1, const int dim2, float* dout );

LITE_API float paddle_FindAbsMax(float* din, int size);

LITE_API float paddle_GetScale(float threshold, int bit_length);

// LITE_API float paddle_fp32_to_int8_1d(const float* din, int8_t* dout, int size);
// LITE_API void paddle_int8_to_fp32_1d(const int8_t* din, float* out, const float scale, int size);
// LITE_API void paddle_int32_to_fp32_1d(const int* din, float* dout, const float scale, int size);
// LITE_API float paddle_int32_to_int8_1d(const int* din, int8_t* dout, const float scale, int size);
// LITE_API float paddle_fp32_to_int16_1d(const float* din, int16_t* dout, int size);
// LITE_API void paddle_int16_to_fp32_1d(const int16_t* din, float* dout, const float scale, int size);





#endif//LITE_WITH_ARM



}  // namespace lite_api
}  // namespace paddle

#endif  // NOLINT
