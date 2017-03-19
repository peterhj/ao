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

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>

__global__ void rect_fwd_kernel_f32(
    uint32_t dim,
    const float *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    float x_i = x[idx];
    y[idx] = x_i * (x_i > 0.0f);
  }
}

extern "C" void arraydiff_cuda_kernel_rect_fwd_f32(
    size_t dim,
    const float *x,
    float *y,
    cudaStream_t stream)
{
  rect_fwd_kernel_f32<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, y);
}

__global__ void rect_bwd_kernel_f32(
    uint32_t dim,
    const float *x,
    const float *dy,
    float *dx)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < dim) {
    dx[idx] = dy[idx] * (x[idx] > 0.0f);
  }
}

extern "C" void arraydiff_cuda_kernel_rect_bwd_f32(
    size_t dim,
    const float *x,
    const float *dy,
    float *dx,
    cudaStream_t stream)
{
  rect_bwd_kernel_f32<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, dy, dx);
}
