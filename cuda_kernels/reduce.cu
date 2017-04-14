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

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void blockreduce_argmax_kernel(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *xs,
    float *x_max,
    uint32_t *x_argmax)
{
  __shared__ float cache[1024 + 32];
  __shared__ uint32_t cache_idx[1024 + 32];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t i = tid + block * block_dim;
  if (tid < block_dim && block < num_blocks) {
    cache[OFFSET_BANK(tid)] = xs[i];
  } else {
    cache[OFFSET_BANK(tid)] = -CUDART_INF_F;
  }
  cache_idx[OFFSET_BANK(tid)] = tid;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if (tid < block_dim && block < num_blocks) {
      if (tid % (2*s) == 0 && (tid + s) < block_dim && cache[OFFSET_BANK(tid)] < cache[OFFSET_BANK(tid + s)]) {
        cache[OFFSET_BANK(tid)]     = cache[OFFSET_BANK(tid + s)];
        cache_idx[OFFSET_BANK(tid)] = cache_idx[OFFSET_BANK(tid + s)];
      }
    }
    __syncthreads();
  }
  if (tid < block_dim && block < num_blocks) {
    if (tid == 0) {
      x_max[block] = cache[0];
      if (NULL != x_argmax) {
        x_argmax[block] = cache_idx[0];
      }
    }
  }
}

extern "C" void arraydiff_cuda_kernel_blockreduce_max_argmax_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *xs,
    float *xs_max,
    uint32_t *xs_argmax,
    cudaStream_t stream)
{
  // XXX: assert(block_dim <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  blockreduce_argmax_kernel<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, xs, xs_max, xs_argmax);
}

/*__global__ void blockreduce_sum_kernel(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *xs,
    float *xs_sum)
{
  __shared__ float cache[1024 + 32];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t i = tid + block * block_dim;
  if (tid < block_dim && block < num_blocks) {
    cache[OFFSET_BANK(tid)] = xs[i];
  } else {
    cache[OFFSET_BANK(tid)] = 0.0f;
  }
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if (tid < block_dim && block < num_blocks) {
      if (tid % (2*s) == 0 && (tid + s) < block_dim) {
        cache[OFFSET_BANK(tid)] += cache[OFFSET_BANK(tid + s)];
      }
    }
    __syncthreads();
  }
  if (tid < block_dim && block < num_blocks) {
    if (tid == 0) {
      xs_sum[block] = cache[0];
    }
  }
}*/

__global__ void blockreduce_sum_kernel(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *xs,
    float *xs_sum)
{
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t idx = tid + block_dim * block;
  if (tid < block_dim && block < num_blocks) {
    cache[tid] = xs[idx];
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (tid < block_dim && block < num_blocks) {
    if (tid == 0) {
      xs_sum[block] = cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_blockreduce_sum_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *xs,
    float *xs_sum,
    cudaStream_t stream)
{
  // XXX: assert(block_dim <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  blockreduce_sum_kernel<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, xs, xs_sum);
}

__global__ void reduce_index_fwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *x,
    const uint32_t *index,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_sz) {
    y[idx] = x[index[idx] + dim * idx];
  }
}

extern "C" void arraydiff_cuda_kernel_reduce_index_fwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *x,
    const uint32_t *index,
    float *y,
    cudaStream_t stream)
{
  size_t n = batch_sz;
  reduce_index_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, x, index, y);
}

__global__ void reduce_index_bwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *dy,
    const uint32_t *index,
    float *dx)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_sz) {
    dx[index[idx] + dim * idx] += dy[idx];
  }
}

extern "C" void arraydiff_cuda_kernel_reduce_index_bwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *dy,
    const uint32_t *index,
    float *dx,
    cudaStream_t stream)
{
  size_t n = batch_sz;
  reduce_index_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, dy, index, dx);
}
