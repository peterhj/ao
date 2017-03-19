/*
Copyright 2017 the arraydiff authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "common.cuh"
#include <cuda_runtime_api.h>
#include <stdint.h>

__global__ void conv_bcast_mult_add_fwd_f32_kernel(
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *scale,
    const float *shift,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t spatial_idx = idx % spatial_dim;
  uint32_t chan_idx = (idx / spatial_dim) % chan_dim;
  uint32_t batch_idx = (idx / spatial_dim) / chan_dim;
  if (spatial_idx < spatial_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] = scale[chan_idx] * x[idx] + shift[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_conv_bcast_mult_add_fwd_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *scale,
    const float *shift,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = spatial_dim * chan_dim * batch_sz;
  conv_bcast_mult_add_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      spatial_dim, chan_dim, batch_sz, x, scale, shift, y);
}

__global__ void conv_bcast_mult_add_param_bwd_nonatomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *scale,
    const float *shift,
    const float *y_grad,
    float *scale_grad,
    float *shift_grad)
{
  __shared__ float cache[1024];
  uint32_t chan_idx = blockIdx.x;
  float scale_value = 0.0f;
  float shift_value = 0.0f;
  if (chan_idx < chan_dim) {
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        float dy = y_grad[idx];
        scale_value += dy * x[idx];
        shift_value += dy;
      }
    }
  }

  cache[threadIdx.x] = scale_value;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim) {
    if (threadIdx.x == 0) {
      scale_grad[chan_idx] += cache[0];
    }
  }
  __syncthreads();

  cache[threadIdx.x] = shift_value;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim) {
    if (threadIdx.x == 0) {
      shift_grad[chan_idx] += cache[0];
    }
  }
}

__global__ void conv_bcast_mult_add_param_bwd_atomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t round_stride,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *scale,
    const float *shift,
    const float *y_grad,
    float *scale_grad,
    float *shift_grad)
{
  __shared__ float cache[1024];
  uint32_t chan_idx = blockIdx.x % chan_dim;
  uint32_t round_start = blockIdx.x / chan_dim;
  float scale_value = 0.0f;
  float shift_value = 0.0f;
  if (chan_idx < chan_dim && round_start < round_stride) {
    for (uint32_t round = round_start; round < num_rounds; round += round_stride) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        float dy = y_grad[idx];
        scale_value += dy * x[idx];
        shift_value += dy;
      }
    }
  }

  cache[threadIdx.x] = scale_value;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim && round_start < round_stride) {
    if (threadIdx.x == 0) {
      atomicAdd(&scale_grad[chan_idx], cache[0]);
    }
  }
  __syncthreads();

  cache[threadIdx.x] = shift_value;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim && round_start < round_stride) {
    if (threadIdx.x == 0) {
      atomicAdd(&shift_grad[chan_idx], cache[0]);
    }
  }
}

extern "C" void arraydiff_cuda_kernel_conv_bcast_mult_add_param_bwd_nonatomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *scale,
    const float *shift,
    const float *y_grad,
    float *scale_grad,
    float *shift_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  conv_bcast_mult_add_param_bwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, spatial_dim, chan_dim, batch_sz, x, scale, shift, y_grad, scale_grad, shift_grad);
}

extern "C" void arraydiff_cuda_kernel_conv_bcast_mult_add_param_bwd_atomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *scale,
    const float *shift,
    const float *y_grad,
    float *scale_grad,
    float *shift_grad,
    cudaStream_t stream)
{
  const uint32_t max_smps = 24; // FIXME: get from cudaDeviceProp.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  if (chan_dim >= min(num_rounds, max_smps)) {
    uint32_t num_blocks = chan_dim;
    conv_bcast_mult_add_param_bwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, spatial_dim, chan_dim, batch_sz, x, scale, shift, y_grad, scale_grad, shift_grad);
  } else {
    uint32_t round_stride = (min(num_rounds, max_smps) + chan_dim - 1) / chan_dim;
    uint32_t num_blocks = round_stride * chan_dim;
    conv_bcast_mult_add_param_bwd_atomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, round_stride, spatial_dim, chan_dim, batch_sz, x, scale, shift, y_grad, scale_grad, shift_grad);
  }
}

__global__ void conv_bcast_mult_add_input_bwd_f32_kernel(
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *scale,
    const float *y_grad,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t spatial_idx = idx % spatial_dim;
  uint32_t chan_idx = (idx / spatial_dim) % chan_dim;
  uint32_t batch_idx = (idx / spatial_dim) / chan_dim;
  if (spatial_idx < spatial_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += y_grad[idx] * scale[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_conv_bcast_mult_add_input_bwd_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *scale,
    const float *y_grad,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = spatial_dim * chan_dim * batch_sz;
  conv_bcast_mult_add_input_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      spatial_dim, chan_dim, batch_sz, scale, y_grad, x_grad);
}
