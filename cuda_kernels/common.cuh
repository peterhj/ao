#ifndef __ARRAYDIFF_CUDA_KERNELS_COMMON_H__
#define __ARRAYDIFF_CUDA_KERNELS_COMMON_H__

#include <stdint.h>

__forceinline__ __device__ void threadblock1024_reduce_sum_f32(
    float *cache)
{
  for (uint32_t s = 512; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      cache[threadIdx.x] += cache[threadIdx.x + s];
    }
    __syncthreads();
  }
}

#endif
