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

__global__ void blockreduce_sum_kernel(
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
