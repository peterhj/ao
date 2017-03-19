/*
COPYRIGHT

All contributions by the University of California:
Copyright (c) 2014-2017 The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014-2017, the respective contributors
All rights reserved.

Caffe uses a shared copyright model: each contributor holds copyright over
their contributions to Caffe. The project versioning records all such
contribution and copyright details. If a contributor wants to further mark
their specific copyright on a particular contribution, they should indicate
their copyright solely in the commit message of the change when it is
committed.

LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CONTRIBUTION AGREEMENT

By contributing to the BVLC/caffe repository through pull-request, comment,
or otherwise, the contributor releases their content to the
license and copyright terms herein.
*/

#include "common.cuh"
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <stdint.h>

__global__ void max_pool_fwd_f32_kernel(
    const uint32_t nthreads,
    const float* const bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    float* const top_data,
    int32_t* const mask)
{
  uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -CUDART_INF_F;
    int maxidx = -1;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    if (NULL != top_data) {
      top_data[index] = maxval;
    }
    if (NULL != mask) {
      mask[index] = maxidx;
    }
  }
}

extern "C" void arraydiff_cuda_kernel_max_pool_fwd_f32(
    size_t x_w, size_t x_h, size_t chan_dim, size_t batch_sz,
    size_t y_w, size_t y_h,
    size_t kernel_w, size_t kernel_h,
    size_t stride_w, size_t stride_h,
    size_t pad_w, size_t pad_h,
    const float *x,
    float *maybe_y,
    int32_t *maybe_mask,
    cudaStream_t stream)
{
  uint32_t n = y_w * y_h * chan_dim * batch_sz;
  max_pool_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, x,
      batch_sz, chan_dim, x_h, x_w,
      y_h, y_w,
      kernel_h, kernel_w,
      stride_h, stride_w,
      pad_h, pad_w,
      maybe_y, maybe_mask);
}

__global__ void max_pool_bwd_f32_kernel(
    const uint32_t nthreads,
    const float* const top_diff,
    const int32_t* const mask,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    float* const bottom_diff)
{
  uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    float gradient = 0.0f;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const float* const top_diff_slice = top_diff + offset;
    const int* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += top_diff_slice[ph * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] += gradient;
  }
}

extern "C" void arraydiff_cuda_kernel_max_pool_bwd_f32(
    size_t x_w, size_t x_h, size_t chan_dim, size_t batch_sz,
    size_t y_w, size_t y_h,
    size_t kernel_w, size_t kernel_h,
    size_t stride_w, size_t stride_h,
    size_t pad_w, size_t pad_h,
    const float *dy,
    const int32_t *mask,
    float *dx,
    cudaStream_t stream)
{
  uint32_t n = x_w * x_h * chan_dim * batch_sz;
  max_pool_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, dy, mask,
      batch_sz, chan_dim, x_h, x_w,
      y_h, y_w,
      kernel_h, kernel_w,
      stride_h, stride_w,
      pad_h, pad_w,
      dx);
}

__global__ void avg_pool_fwd_f32_kernel(
    const uint32_t nthreads,
    const float* const bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    float* const top_data)
{
  uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    float aveval = 0.0f;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

extern "C" void arraydiff_cuda_kernel_avg_pool_fwd_f32(
    size_t x_w, size_t x_h, size_t chan_dim, size_t batch_sz,
    size_t y_w, size_t y_h,
    size_t kernel_w, size_t kernel_h,
    size_t stride_w, size_t stride_h,
    size_t pad_w, size_t pad_h,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = y_w * y_h * chan_dim * batch_sz;
  avg_pool_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, x,
      batch_sz, chan_dim, x_h, x_w,
      y_h, y_w,
      kernel_h, kernel_w,
      stride_h, stride_w,
      pad_h, pad_w,
      y);
}

__global__ void avg_pool_bwd_f32_kernel(
    const uint32_t nthreads,
    const float* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    float* const bottom_diff)
{
  uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    float gradient = 0.0f;
    const float* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] += gradient;
  }
}

extern "C" void arraydiff_cuda_kernel_avg_pool_bwd_f32(
    size_t x_w, size_t x_h, size_t chan_dim, size_t batch_sz,
    size_t y_w, size_t y_h,
    size_t kernel_w, size_t kernel_h,
    size_t stride_w, size_t stride_h,
    size_t pad_w, size_t pad_h,
    const float *dy,
    float *dx,
    cudaStream_t stream)
{
  uint32_t n = x_w * x_h * chan_dim * batch_sz;
  avg_pool_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      n, dy,
      batch_sz, chan_dim, x_h, x_w,
      y_h, y_w,
      kernel_h, kernel_w,
      stride_h, stride_w,
      pad_h, pad_w,
      dx);
}
