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
#include <math_constants.h>
#include <stdint.h>
#include <stdlib.h>

//#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) & 31)); })

__global__ void block_softmax_fwd_f32(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *x,
    float *y)
{
  __shared__ float cache[1024];
  //__shared__ uint32_t cache_idx[1024];
  //__shared__ float result[1];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t i = tid + block * block_dim;

  float x_i = 0.0f;
  if (tid < block_dim && block < num_blocks) {
    x_i = x[i];
    cache[tid] = x_i;
  } else {
    cache[tid] = -CUDART_INF_F;
  }
  __syncthreads();
  /*for (uint32_t s = 1; s < 1024; s *= 2) {
    if (tid < block_dim && block < num_blocks) {
      if ((tid & (2 * s - 1)) == 0 && (tid + s) < block_dim) {
        if (cache[OFFSET_BANK(tid)] < cache[OFFSET_BANK(tid + s)]) {
          cache[OFFSET_BANK(tid)] = cache[OFFSET_BANK(tid + s)];
        }
      }
    }
    __syncthreads();
  }*/
  threadblock1024_reduce_max_f32(cache);
  float max_logit = cache[0];
  __syncthreads();

  float z_i = 0.0f;
  if (tid < block_dim && block < num_blocks) {
    z_i = expf(x_i - max_logit);
    cache[tid] = z_i;
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  /*for (uint32_t s = 1; s < 1024; s *= 2) {
    if (tid < block_dim && block < num_blocks) {
      if ((tid & (2 * s - 1)) == 0 && (tid + s) < block_dim) {
        cache[OFFSET_BANK(tid)] += cache[OFFSET_BANK(tid + s)];
      }
    }
    __syncthreads();
  }*/
  threadblock1024_reduce_sum_f32(cache);
  float sum_factor = cache[0];
  __syncthreads();

  if (tid < block_dim && block < num_blocks) {
    y[i] = z_i / sum_factor;
  }
}

extern "C" void arraydiff_cuda_kernel_block_softmax_fwd_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  // XXX: assert(block_dim <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  block_softmax_fwd_f32<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, x, y);
}

__global__ void softmax_nll_loss_fwd_f32(
    uint32_t dim,
    uint32_t batch_sz,
    const float *y,
    const uint32_t *t,
    float *loss)
{
  uint32_t batch_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (batch_idx < batch_sz) {
    uint32_t offset_i = t[batch_idx];
    uint32_t idx = offset_i + dim * batch_idx;
    loss[batch_idx] = -logf(y[idx]);
  }
}

extern "C" void arraydiff_cuda_kernel_softmax_nll_loss_fwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *y,
    const uint32_t *t,
    float *loss,
    cudaStream_t stream)
{
  softmax_nll_loss_fwd_f32<<<(batch_sz+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, y, t, loss);
}

__global__ void softmax_nll_loss_bwd_f32(
    uint32_t dim,
    uint32_t batch_sz,
    const float *y,
    const uint32_t *t,
    const float *df,
    float *dx)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t i = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (i < dim && batch_idx < batch_sz) {
    uint32_t t_i = t[batch_idx];
    dx[idx] += df[batch_idx] * (y[idx] - (float)(i == t_i));
  }
}

extern "C" void arraydiff_cuda_kernel_softmax_nll_loss_bwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *y,
    const uint32_t *t,
    const float *df,
    float *dx,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  softmax_nll_loss_bwd_f32<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, y, t, df, dx);
}

__global__ void block_softmax_negentropy_loss_fwd_accumulate_f32(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *y,
    float *loss)
{
  const float beta = 0.01f;
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t i = tid + block * block_dim;

  float y_i = 0.0f;
  float ent_i = 0.0f;
  if (tid < block_dim && block < num_blocks) {
    y_i = y[i];
    if (y_i > 0.0f) {
      ent_i = -y_i * logf(y_i);
    }
    cache[tid] = ent_i;
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);

  if (tid == 0 && block < num_blocks) {
    float entropy = cache[0];
    loss[block] += -beta * entropy;
  }
}

extern "C" void arraydiff_cuda_kernel_block_softmax_negentropy_loss_fwd_accumulate_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *y,
    float *loss,
    cudaStream_t stream)
{
  block_softmax_negentropy_loss_fwd_accumulate_f32<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, y, loss);
}

__global__ void block_softmax_negentropy_loss_bwd_f32(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *y,
    const float *df,
    float *dx)
{
  const float beta = 0.01f;
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t i = tid + block * block_dim;

  float y_i = 0.0f;
  float ent_i = 0.0f;
  if (tid < block_dim && block < num_blocks) {
    y_i = y[i];
    if (y_i > 0.0f) {
      ent_i = -y_i * logf(y_i);
    }
    cache[tid] = ent_i;
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);

  if (tid < block_dim && block < num_blocks) {
    float entropy = cache[0];
    float diff_i = -beta * (ent_i - y_i * entropy);
    dx[i] += df[block] * diff_i;
  }
}

extern "C" void arraydiff_cuda_kernel_block_softmax_negentropy_loss_bwd_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *y,
    const float *df,
    float *dx,
    cudaStream_t stream)
{
  block_softmax_negentropy_loss_bwd_f32<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, y, df, dx);
}
