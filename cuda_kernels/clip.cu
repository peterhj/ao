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
#include <stdlib.h>

__global__ void symm_unit_clip_fwd_f32_kernel(
    uint32_t dim,
    const float *clip,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    /*float c = clip[0];
    float a = c / max(fabs(c), 1.0f);*/
    float a = clip[0];
    float x_i = x[idx];
    y[idx] = x_i * ((x_i > 0.0f) + a * (x_i < 0.0f));
  }
}

extern "C" void arraydiff_cuda_kernel_symm_unit_clip_fwd_f32(
    size_t dim,
    const float *clip,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  symm_unit_clip_fwd_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, clip, x, y);
}

__global__ void symm_unit_clip_param_bwd_f32_atomic_naive_kernel(
    uint32_t dim,
    const float *clip,
    const float *x,
    const float *dy,
    float *grad)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    /*float c = clip[0];
    float u = max(fabs(c), 1.0f);
    float du = 1.0f * (c > 1.0f) - 1.0f * (c < -1.0f);
    float x_i = x[idx];
    atomicAdd(&grad[0], (1.0f / u) * (1.0f - du * c / u) * dy[idx] * x_i * (x_i < 0.0f));*/
    float x_i = x[idx];
    atomicAdd(&grad[0], dy[idx] * x_i * (x_i < 0.0f));
  }
}

__global__ void symm_unit_clip_param_bwd_f32_atomic_fast_kernel(
    uint32_t dim,
    const float *clip,
    const float *x,
    const float *dy,
    float *grad)
{
  __shared__ float cache[1024];
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    /*float c = clip[0];
    float u = max(fabs(c), 1.0f);
    float du = 1.0f * (c > 1.0f) - 1.0f * (c < -1.0f);
    float x_i = x[idx];
    cache[threadIdx.x] = (1.0f / u) * (1.0f - du * c / u) * dy[idx] * x_i * (x_i < 0.0f);*/
    float x_i = x[idx];
    cache[threadIdx.x] = dy[idx] * x_i * (x_i < 0.0f);
  } else {
    cache[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  threadblock1024_reduce_sum_f32(cache);
  if (idx < dim) {
    if (threadIdx.x == 0) {
      atomicAdd(&grad[0], cache[0]);
    }
  }
}

extern "C" void arraydiff_cuda_kernel_symm_unit_clip_param_bwd_nondeterministic_f32(
    size_t dim,
    const float *clip,
    const float *x,
    const float *dy,
    float *grad,
    cudaStream_t stream)
{
  symm_unit_clip_param_bwd_f32_atomic_fast_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, clip, x, dy, grad);
}

__global__ void symm_unit_clip_input_bwd_f32_kernel(
    uint32_t dim,
    const float *clip,
    const float *x,
    const float *dy,
    float *dx)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    /*float c = clip[0];
    float a = c / max(fabs(c), 1.0f);*/
    float a = clip[0];
    float x_i = x[idx];
    dx[idx] += dy[idx] * ((x_i > 0.0f) + a * (x_i < 0.0f));
  }
}

extern "C" void arraydiff_cuda_kernel_symm_unit_clip_input_bwd_f32(
    size_t dim,
    const float *clip,
    const float *x,
    const float *dy,
    float *dx,
    cudaStream_t stream)
{
  symm_unit_clip_input_bwd_f32_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, clip, x, dy, dx);
}
