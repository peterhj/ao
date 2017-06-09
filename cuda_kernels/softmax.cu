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

__global__ void block_softmax_fwd_f32(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *x,
    float *y)
{
  __shared__ float cache[1024];
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

__global__ void block_softmax_tangent_fwd_f32(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *px,
    const float *x,
    const float *py,
    float *y)
{
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t idx = tid + block_dim * block;

  float x_i = 0.0f;
  if (tid < block_dim && block < num_blocks) {
    x_i = px[idx];
    cache[tid] = x_i;
  } else {
    cache[tid] = -CUDART_INF_F;
  }
  __syncthreads();
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
  threadblock1024_reduce_sum_f32(cache);
  float sum_factor = cache[0];
  __syncthreads();

  if (tid < block_dim && block < num_blocks) {
    cache[tid] = z_i * x_i;
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  float weighted_sum_factor = cache[0];
  __syncthreads();

  if (tid < block_dim && block < num_blocks) {
    y[idx] = py[idx] * (x[idx] - weighted_sum_factor / sum_factor);
  }
}

extern "C" void arraydiff_cuda_kernel_block_softmax_tangent_fwd_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *px,
    const float *x,
    const float *py,
    float *y,
    cudaStream_t stream)
{
  // XXX: assert(block_dim <= 1024);
  // FIXME(20151022): could make more efficient use of blocks but w/e.
  block_softmax_tangent_fwd_f32<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, px, x, py, y);
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

__global__ void block_softmax_kl2_loss_fwd_f32_kernel(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *y,
    const float *t,
    float *loss)
{
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t idx = tid + block_dim * block;

  if (tid < block_dim && block < num_blocks) {
    float y_i = y[idx];
    float t_i = t[idx];
    float kl_i = 0.0f;
    if (t_i > 0.0f) {
      kl_i = t_i * (logf(t_i) - logf(y_i));
    } else {
      kl_i = -t_i * logf(y_i);
    }
    cache[tid] = kl_i;
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_max_f32(cache);

  if (tid < block_dim && block < num_blocks) {
    if (tid == 0) {
      loss[block] = cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_block_softmax_kl2_loss_fwd_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *y,
    const float *t,
    float *loss,
    cudaStream_t stream)
{
  block_softmax_kl2_loss_fwd_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, y, t, loss);
}

__global__ void softmax_kl2_loss_bwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *y,
    const float *t,
    const float *df,
    float *dx)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t j = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (j < dim && batch_idx < batch_sz) {
    float y_i = y[idx];
    float t_i = t[idx];
    dx[idx] += df[batch_idx] * (y_i - t_i);
  }
}

extern "C" void arraydiff_cuda_kernel_softmax_kl2_loss_bwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *y,
    const float *t,
    const float *df,
    float *dx,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  softmax_kl2_loss_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, y, t, df, dx);
}

__global__ void block_softmax_tangent_kl2_loss_fwd_f32_kernel(
    uint32_t block_dim,
    uint32_t num_blocks,
    const float *py,
    const float *y,
    const float *t,
    float *loss)
{
  __shared__ float cache[1024];
  uint32_t tid = threadIdx.x;
  uint32_t block = blockIdx.x;
  uint32_t idx = tid + block_dim * block;

  if (tid < block_dim && block < num_blocks) {
    float py_i = py[idx];
    float y_i = y[idx];
    float t_i = t[idx];
    cache[tid] = -t_i * y_i / py_i;
  } else {
    cache[tid] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_max_f32(cache);

  if (tid < block_dim && block < num_blocks) {
    if (tid == 0) {
      loss[block] = cache[0];
    }
  }
}

extern "C" void arraydiff_cuda_kernel_block_softmax_tangent_kl2_loss_fwd_f32(
    size_t block_dim,
    size_t num_blocks,
    const float *py,
    const float *y,
    const float *t,
    float *loss,
    cudaStream_t stream)
{
  block_softmax_tangent_kl2_loss_fwd_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      block_dim, num_blocks, py, y, t, loss);
}

__global__ void softmax_tangent_kl2_loss_bwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *y,
    const float *df,
    float *dx)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t j = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (j < dim && batch_idx < batch_sz) {
    float y_i = y[idx];
    dx[idx] += df[batch_idx] * y_i;
  }
}

extern "C" void arraydiff_cuda_kernel_softmax_tangent_kl2_loss_bwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *y,
    const float *df,
    float *dx,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  softmax_tangent_kl2_loss_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, y, df, dx);
}

__global__ void softmax_lr_loss_fwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *y,
    const uint32_t *index,
    const float *t,
    float *loss,
    float lr_clip)
{
  uint32_t batch_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (batch_idx < batch_sz) {
    uint32_t index_i = index[batch_idx];
    float y_i = y[index_i + dim * batch_idx];
    float t_i = t[batch_idx];
    float lr_i = y_i / t_i;
    loss[batch_idx] = min(lr_i, lr_clip);
  }
}

extern "C" void arraydiff_cuda_kernel_softmax_lr_loss_fwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *y,
    const uint32_t *index,
    const float *t,
    float *loss,
    float lr_clip,
    cudaStream_t stream)
{
  softmax_lr_loss_fwd_f32_kernel<<<(batch_sz + 1024 - 1) / 1024, 1024, 0, stream>>>(
      dim, batch_sz, y, index, t, loss, lr_clip);
}

__global__ void softmax_lr_loss_bwd_f32_kernel(
    uint32_t dim,
    uint32_t batch_sz,
    const float *y,
    const uint32_t *index,
    const float *t,
    const float *df,
    float *dx,
    float lr_clip)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t j = idx % dim;
  uint32_t batch_idx = idx / dim;
  if (j < dim && batch_idx < batch_sz) {
    uint32_t index_i = index[batch_idx];
    float delta_i = (float)(j == index_i);
    float y_ij = y[idx];
    float y_i = y[index_i + dim * batch_sz];
    float t_i = t[batch_idx];
    float lr_i = y_i / t_i;
    if (lr_i < lr_clip) {
      dx[idx] += df[batch_idx] * lr_i * (delta_i - y_ij);
    }
  }
}

extern "C" void arraydiff_cuda_kernel_softmax_lr_loss_bwd_f32(
    size_t dim,
    size_t batch_sz,
    const float *y,
    const uint32_t *index,
    const float *t,
    const float *df,
    float *dx,
    float lr_clip,
    cudaStream_t stream)
{
  uint32_t n = dim * batch_sz;
  softmax_lr_loss_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      dim, batch_sz, y, index, t, df, dx, lr_clip);
}
