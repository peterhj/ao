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

__global__ void block_lst_sq_fwd_f32_kernel(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *x,
    const float *target,
    float *loss,
    uint32_t do_clip)
{
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t idx = tid + block_dim * block;

  if (tid < block_dim && block < num_blocks) {
    float x_i = x[idx];
    float t_i = target[idx];
    float delta = x_i - t_i;
    if (do_clip) {
      if (fabs(delta) > 1.0f) {
        cache[tid] = fabs(delta);
      } else {
        cache[tid] = 0.5f * delta * delta;
      }
    } else {
      cache[tid] = 0.5f * delta * delta;
    }
  } else {
    cache[tid] = -CUDART_INF_F;
  }
  __syncthreads();

  threadblock1024_reduce_sum_f32(cache);
  if (tid < block_dim && block < num_blocks) {
    loss[block] = cache[0];
  }
}

extern "C" void arraydiff_cuda_kernel_block_lst_sq_fwd_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *x,
    const float *target,
    float *loss,
    uint32_t do_clip,
    cudaStream_t stream)
{
  block_lst_sq_fwd_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, x, target, loss, do_clip);
}

__global__ void block_lst_sq_bwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *x,
    const float *target,
    const float *df,
    float *dx,
    uint32_t do_clip)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t i = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (i < dim && batch_idx < batch_sz) {
    float delta = x[idx] - target[idx];
    if (do_clip) {
      float clipped_delta = max(-1.0f, min(delta, 1.0f));
      dx[idx] += df[batch_idx] * clipped_delta;
    } else {
      dx[idx] += df[batch_idx] * delta;
    }
  }
}

extern "C" void arraydiff_cuda_kernel_block_lst_sq_bwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *x,
    const float *target,
    const float *df,
    float *dx,
    uint32_t do_clip,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  block_lst_sq_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, x, target, df, dx, do_clip);
}
