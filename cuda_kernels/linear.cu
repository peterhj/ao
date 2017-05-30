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

/* Broadcast add kernels: [a] . [an] -> [an] . */

__global__ void bcast_add_I1a_I2an_O1an_fwd_f32_kernel(
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *shift,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t chan_idx = idx % chan_dim;
  uint32_t batch_idx = idx / chan_dim;
  if (chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] = x[idx] + shift[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2an_O1an_fwd_f32(
    size_t chan_dim,
    size_t batch_sz,
    const float *shift,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = chan_dim * batch_sz;
  bcast_add_I1a_I2an_O1an_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      chan_dim, batch_sz, shift, x, y);
}

__global__ void bcast_add_I1a_I2an_O1an_fwdaccum_f32_kernel(
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *shift,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t chan_idx = idx % chan_dim;
  uint32_t batch_idx = idx / chan_dim;
  if (chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] += x[idx] + shift[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2an_O1an_fwdaccum_f32(
    size_t chan_dim,
    size_t batch_sz,
    const float *shift,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = chan_dim * batch_sz;
  bcast_add_I1a_I2an_O1an_fwdaccum_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      chan_dim, batch_sz, shift, x, y);
}

__global__ void bcast_add_I1a_I2an_O1an_bwd_shift_deterministic_f32_kernel(
    uint32_t num_rounds,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *y_grad,
    float *shift_grad)
{
  __shared__ float cache[1024];
  uint32_t chan_idx = blockIdx.x;
  float shift_grad_acc = 0.0f;
  if (chan_idx < chan_dim) {
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t batch_idx = round_idx;
      if (batch_idx < batch_sz) {
        uint32_t idx = chan_idx + chan_dim * batch_idx;
        float dy = y_grad[idx];
        shift_grad_acc += dy;
      }
    }
  }

  cache[threadIdx.x] = shift_grad_acc;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim) {
    if (threadIdx.x == 0) {
      shift_grad[blockIdx.x] += cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2an_O1an_bwd_shift_deterministic_f32(
    size_t chan_dim,
    size_t batch_sz,
    const float *y_grad,
    float *shift_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  bcast_add_I1a_I2an_O1an_bwd_shift_deterministic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, chan_dim, batch_sz, y_grad, shift_grad);
}

__global__ void bcast_add_I1a_I2an_O1an_bwd_input_f32_kernel(
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *y_grad,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t chan_idx = idx % chan_dim;
  uint32_t batch_idx = idx / chan_dim;
  if (chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += y_grad[idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2an_O1an_bwd_input_f32(
    size_t chan_dim,
    size_t batch_sz,
    const float *y_grad,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = chan_dim * batch_sz;
  bcast_add_I1a_I2an_O1an_bwd_input_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      chan_dim, batch_sz, y_grad, x_grad);
}

/* Broadcast add kernels: [a] . [xyan] -> [xyan] . */

__global__ void bcast_add_I1a_I2xyan_O1xyan_fwd_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *shift,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] = x[idx] + shift[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2xyan_O1xyan_fwd_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *shift,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_add_I1a_I2xyan_O1xyan_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, shift, x, y);
}

__global__ void bcast_add_I1a_I2xyan_O1xyan_fwdaccum_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *shift,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] += x[idx] + shift[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2xyan_O1xyan_fwdaccum_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *shift,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_add_I1a_I2xyan_O1xyan_fwdaccum_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, shift, x, y);
}

__global__ void bcast_add_I1a_I2xyan_O1xyan_bwd_shift_deterministic_f32_kernel(
    uint32_t num_rounds,
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *y_grad,
    float *shift_grad)
{
  __shared__ float cache[1024];
  uint32_t chan_idx = blockIdx.x;
  float shift_grad_acc = 0.0f;
  if (chan_idx < chan_dim) {
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t prefix_idx = round_idx % prefix_dim;
      uint32_t batch_idx = round_idx / prefix_dim;
      if (prefix_idx < prefix_dim && batch_idx < batch_sz) {
        uint32_t idx = prefix_idx + prefix_dim * (chan_idx + chan_dim * batch_idx);
        float dy = y_grad[idx];
        shift_grad_acc += dy;
      }
    }
  }

  cache[threadIdx.x] = shift_grad_acc;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim) {
    if (threadIdx.x == 0) {
      shift_grad[blockIdx.x] += cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2xyan_O1xyan_bwd_shift_deterministic_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *y_grad,
    float *shift_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (prefix_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  bcast_add_I1a_I2xyan_O1xyan_bwd_shift_deterministic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, prefix_dim, chan_dim, batch_sz, y_grad, shift_grad);
}

__global__ void bcast_add_I1a_I2xyan_O1xyan_bwd_input_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *y_grad,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += y_grad[idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1a_I2xyan_O1xyan_bwd_input_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *y_grad,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_add_I1a_I2xyan_O1xyan_bwd_input_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, y_grad, x_grad);
}

/* Broadcast multiply-add kernels: [a] . [a] . [xyan] -> [xyan]. */

__global__ void bcast_mult_add_I1a_I2a_I3xyan_O1xyan_fwd_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *scale,
    const float *shift,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] = scale[chan_idx] * x[idx] + shift[chan_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_mult_add_I1a_I2a_I3xyan_O1xyan_fwd_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *scale,
    const float *shift,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_mult_add_I1a_I2a_I3xyan_O1xyan_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, scale, shift, x, y);
}

__global__ void bcast_mult_add_I1a_I2a_I3xyan_O1xyan_bwd_scale_shift_deterministic_f32_kernel(
    uint32_t num_rounds,
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *y_grad,
    float *scale_grad,
    float *shift_grad)
{
  __shared__ float cache[1024];
  uint32_t chan_idx = blockIdx.x;
  float scale_grad_acc = 0.0f;
  float shift_grad_acc = 0.0f;
  if (chan_idx < chan_dim) {
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t prefix_idx = round_idx % prefix_dim;
      uint32_t batch_idx = round_idx / prefix_dim;
      if (prefix_idx < prefix_dim && batch_idx < batch_sz) {
        uint32_t idx = prefix_idx + prefix_dim * (chan_idx + chan_dim * batch_idx);
        float dy = y_grad[idx];
        scale_grad_acc += dy * x[idx];
        shift_grad_acc += dy;
      }
    }
  }

  cache[threadIdx.x] = scale_grad_acc;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim) {
    if (threadIdx.x == 0) {
      scale_grad[blockIdx.x] += cache[0];
    }
  }
  __syncthreads();

  cache[threadIdx.x] = shift_grad_acc;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim) {
    if (threadIdx.x == 0) {
      shift_grad[blockIdx.x] += cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_mult_add_I1a_I2a_I3xyan_O1xyan_bwd_shift_deterministic_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *y_grad,
    float *scale_grad,
    float *shift_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (prefix_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  bcast_mult_add_I1a_I2a_I3xyan_O1xyan_bwd_scale_shift_deterministic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, prefix_dim, chan_dim, batch_sz, x, y_grad, scale_grad, shift_grad);
}

__global__ void bcast_mult_add_I1a_I2a_I3xyan_O1xyan_bwd_input_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *scale,
    const float *y_grad,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += scale[chan_idx] * y_grad[idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_mult_add_I1a_I2a_I3xyan_O1xyan_bwd_input_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *scale,
    const float *y_grad,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_mult_add_I1a_I2a_I3xyan_O1xyan_bwd_input_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, scale, y_grad, x_grad);
}

/* Broadcast add kernels: [an] . [xyan] -> [xyan]. */

__global__ void bcast_add_I1an_I2xyan_O1xyan_fwd_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *shift,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  uint32_t shift_idx = chan_idx + chan_dim * batch_idx;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] = x[idx] + shift[shift_idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1an_I2xyan_O1xyan_fwd_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *shift,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_add_I1an_I2xyan_O1xyan_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, shift, x, y);
}

__global__ void bcast_add_I1an_I2xyan_O1xyan_bwd_shift_deterministic_f32_kernel(
    uint32_t num_rounds,
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *y_grad,
    float *shift_grad)
{
  __shared__ float cache[1024];
  uint32_t chan_idx = blockIdx.x % chan_dim;
  uint32_t batch_idx = blockIdx.x / chan_dim;
  float shift_grad_acc = 0.0f;
  if (chan_idx < chan_dim && batch_idx < batch_sz) {
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      //uint32_t block_dim = min(blockDim.x, prefix_dim - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t prefix_idx = round_idx;
      if (prefix_idx < prefix_dim) {
        uint32_t idx = prefix_idx + prefix_dim * (chan_idx + chan_dim * batch_idx);
        float dy = y_grad[idx];
        shift_grad_acc += dy;
      }
    }
  }

  cache[threadIdx.x] = shift_grad_acc;
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (chan_idx < chan_dim && batch_idx < batch_sz) {
    if (threadIdx.x == 0) {
      shift_grad[blockIdx.x] += cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1an_I2xyan_O1xyan_bwd_shift_deterministic_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *y_grad,
    float *shift_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (prefix_dim + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim * batch_sz;
  bcast_add_I1an_I2xyan_O1xyan_bwd_shift_deterministic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, prefix_dim, chan_dim, batch_sz, y_grad, shift_grad);
}

__global__ void bcast_add_I1an_I2xyan_O1xyan_bwd_input_f32_kernel(
    uint32_t prefix_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *y_grad,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t prefix_idx = idx % prefix_dim;
  uint32_t chan_idx = (idx / prefix_dim) % chan_dim;
  uint32_t batch_idx = (idx / prefix_dim) / chan_dim;
  if (prefix_idx < prefix_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += y_grad[idx];
  }
}

extern "C" void arraydiff_cuda_kernel_bcast_add_I1an_I2xyan_O1xyan_bwd_input_f32(
    size_t prefix_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *y_grad,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = prefix_dim * chan_dim * batch_sz;
  bcast_add_I1an_I2xyan_O1xyan_bwd_input_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      prefix_dim, chan_dim, batch_sz, y_grad, x_grad);
}
