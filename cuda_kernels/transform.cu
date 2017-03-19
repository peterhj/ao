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

__global__ void cast_u8_to_f32(
    uint32_t dim,
    const uint8_t *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    y[idx] = (float)(x[idx]);
  }
}

extern "C" void arraydiff_cuda_kernel_cast_u8_to_f32(
    size_t dim,
    const uint8_t *x,
    float *y,
    cudaStream_t stream)
{
  cast_u8_to_f32<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, y);
}

__global__ void cast_u8x4_to_f32x4(
    uint32_t dim,
    const uint8_t *x,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx + 4 <= dim) {
    uint32_t i = idx >> 2;
    uchar4 vx_i = ((const uchar4 *)x)[i];
    float4 vy_i = { (float)vx_i.x, (float)vx_i.y, (float)vx_i.z, (float)vx_i.w };
    ((float4 *)y)[i] = vy_i;
  } else if (idx < dim) {
    y[idx] = (float)(x[idx]);
  }
}

extern "C" void arraydiff_cuda_kernel_cast_u8x4_to_f32x4(
    size_t dim,
    const uint8_t *x,
    float *y,
    cudaStream_t stream)
{
  cast_u8x4_to_f32x4<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, y);
}
