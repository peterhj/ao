#include <cuda_runtime_api.h>
#include <stdint.h>

#define OFFSET_BANK(idx) ({ __typeof__ (idx) _idx = idx; ((_idx) + ((_idx) / 32)); })

__global__ void conv_normalize_fwd_f32_kernel(
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    float epsilon,
    float *y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t spatial_idx = idx % spatial_dim;
  uint32_t chan_idx = (idx / spatial_dim) % chan_dim;
  //uint32_t batch_idx = idx / (spatial_dim * chan_dim);
  uint32_t batch_idx = (idx / spatial_dim) / chan_dim;
  if (spatial_idx < spatial_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    y[idx] = (x[idx] - mean[chan_idx]) * rsqrtf(var[chan_idx] + epsilon);
  }
}

extern "C" void arraydiff_cuda_kernel_conv_normalize_fwd_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    float epsilon,
    float *y,
    cudaStream_t stream)
{
  uint32_t n = spatial_dim * chan_dim * batch_sz;
  conv_normalize_fwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      spatial_dim, chan_dim, batch_sz, x, mean, var, epsilon, y);
}

__global__ void conv_normalize_var_bwd_nonatomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *y_grad,
    float epsilon,
    float *var_grad)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x;
  float value = 0.0f;
  if (chan_idx < chan_dim) {
    float m = mean[chan_idx];
    float v = var[chan_idx];
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        value += -0.5f * y_grad[idx] * (x[idx] - m) * rsqrtf(v + epsilon) / (v + epsilon);
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    var_grad[chan_idx] += cache[0];
  }
}

__global__ void conv_normalize_var_bwd_atomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t round_stride,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *y_grad,
    float epsilon,
    float *var_grad)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x % chan_dim;
  uint32_t round_start = blockIdx.x / chan_dim;
  float value = 0.0f;
  if (chan_idx < chan_dim && round_start < round_stride) {
    float m = mean[chan_idx];
    float v = var[chan_idx];
    for (uint32_t round = round_start; round < num_rounds; round += round_stride) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        value += -0.5f * y_grad[idx] * (x[idx] - m) * rsqrtf(v + epsilon) / (v + epsilon);
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&var_grad[chan_idx], cache[0]);
  }
}

extern "C" void arraydiff_cuda_kernel_conv_normalize_var_bwd_nonatomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *y_grad,
    float epsilon,
    float *var_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  conv_normalize_var_bwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, spatial_dim, chan_dim, batch_sz, x, mean, var, y_grad, epsilon, var_grad);
}

extern "C" void arraydiff_cuda_kernel_conv_normalize_var_bwd_atomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *y_grad,
    float epsilon,
    float *var_grad,
    cudaStream_t stream)
{
  const uint32_t max_smps = 24; // FIXME: get from cudaDeviceProp.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  if (chan_dim >= min(num_rounds, max_smps)) {
    uint32_t num_blocks = chan_dim;
    conv_normalize_var_bwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, spatial_dim, chan_dim, batch_sz, x, mean, var, y_grad, epsilon, var_grad);
  } else {
    uint32_t round_stride = (min(num_rounds, max_smps) + chan_dim - 1) / chan_dim;
    uint32_t num_blocks = round_stride * chan_dim;
    conv_normalize_var_bwd_atomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, round_stride, spatial_dim, chan_dim, batch_sz, x, mean, var, y_grad, epsilon, var_grad);
  }
}

__global__ void conv_normalize_mean_bwd_nonatomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *var_grad,
    const float *y_grad,
    float epsilon,
    float *mean_grad)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x;
  float value = 0.0f;
  if (chan_idx < chan_dim) {
    float m = mean[chan_idx];
    float v = var[chan_idx];
    float dv = var_grad[chan_idx];
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        value += -(y_grad[idx] * rsqrtf(v + epsilon) + 2.0f * dv * (x[idx] - m) / ((float)(spatial_dim * (batch_sz - 1))));
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    mean_grad[chan_idx] += cache[0];
  }
}

__global__ void conv_normalize_mean_bwd_atomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t round_stride,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *var_grad,
    const float *y_grad,
    float epsilon,
    float *mean_grad)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x % chan_dim;
  uint32_t round_start = blockIdx.x / chan_dim;
  float value = 0.0f;
  if (chan_idx < chan_dim && round_start < round_stride) {
    float m = mean[chan_idx];
    float v = var[chan_idx];
    float dv = var_grad[chan_idx];
    for (uint32_t round = round_start; round < num_rounds; round += round_stride) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        value += -(y_grad[idx] * rsqrtf(v + epsilon) + 2.0f * dv * (x[idx] - m) / ((float)(spatial_dim * (batch_sz - 1))));
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&mean_grad[chan_idx], cache[0]);
  }
}

extern "C" void arraydiff_cuda_kernel_conv_normalize_mean_bwd_nonatomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *var_grad,
    const float *y_grad,
    float epsilon,
    float *mean_grad,
    cudaStream_t stream)
{
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  conv_normalize_mean_bwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, spatial_dim, chan_dim, batch_sz, x, mean, var, var_grad, y_grad, epsilon, mean_grad);
}

extern "C" void arraydiff_cuda_kernel_conv_normalize_mean_bwd_atomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    const float *var,
    const float *var_grad,
    const float *y_grad,
    float epsilon,
    float *mean_grad,
    cudaStream_t stream)
{
  const uint32_t max_smps = 24; // FIXME: get from cudaDeviceProp.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  if (chan_dim >= min(num_rounds, max_smps)) {
    uint32_t num_blocks = chan_dim;
    conv_normalize_mean_bwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, spatial_dim, chan_dim, batch_sz, x, mean, var, var_grad, y_grad, epsilon, mean_grad);
  } else {
    uint32_t round_stride = (min(num_rounds, max_smps) + chan_dim - 1) / chan_dim;
    uint32_t num_blocks = round_stride * chan_dim;
    conv_normalize_mean_bwd_atomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, round_stride, spatial_dim, chan_dim, batch_sz, x, mean, var, var_grad, y_grad, epsilon, mean_grad);
  }
}

__global__ void conv_normalize_input_bwd_nonatomic_f32_kernel(
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *var,
    const float *y_grad,
    float epsilon,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t spatial_idx = idx % spatial_dim;
  uint32_t chan_idx = (idx / spatial_dim) % chan_dim;
  uint32_t batch_idx = (idx / spatial_dim) / chan_dim;
  if (spatial_idx < spatial_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += y_grad[idx] * rsqrtf(var[chan_idx] + epsilon);
  }
}

extern "C" void arraydiff_cuda_kernel_conv_normalize_input_bwd_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *var,
    const float *y_grad,
    float epsilon,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = spatial_dim * chan_dim * batch_sz;
  conv_normalize_input_bwd_nonatomic_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      spatial_dim, chan_dim, batch_sz, var, y_grad, epsilon, x_grad);
}

__global__ void conv_batch_stats_mean_fwd_nonatomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    float *mean)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x;
  float value = 0.0f;
  if (chan_idx < chan_dim) {
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        value += x[idx];
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    mean[chan_idx] += cache[0] / ((float)(spatial_dim * batch_sz));
  }
}

__global__ void conv_batch_stats_mean_fwd_atomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t round_stride,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    float *mean)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x % chan_dim;
  uint32_t round_start = blockIdx.x / chan_dim;
  float value = 0.0f;
  if (chan_idx < chan_dim && round_start < round_stride) {
    for (uint32_t round = round_start; round < num_rounds; round += round_stride) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t block_dim = min(blockDim.x, spatial_dim * batch_sz - round_offset);
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        value += x[idx];
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&mean[chan_idx], cache[0] / ((float)(spatial_dim * batch_sz)));
  }
}

extern "C" void arraydiff_cuda_kernel_conv_batch_stats_mean_fwd_nonatomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    float *mean,
    cudaStream_t stream)
{
  // XXX: `mean` should be zeroed.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  conv_batch_stats_mean_fwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, spatial_dim, chan_dim, batch_sz, x, mean);
}

extern "C" void arraydiff_cuda_kernel_conv_batch_stats_mean_fwd_atomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    float *mean,
    cudaStream_t stream)
{
  // XXX: `mean` should be zeroed.
  const uint32_t max_smps = 24; // FIXME: get from cudaDeviceProp.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  if (chan_dim >= min(num_rounds, max_smps)) {
    uint32_t num_blocks = chan_dim;
    conv_batch_stats_mean_fwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, spatial_dim, chan_dim, batch_sz, x, mean);
  } else {
    uint32_t round_stride = (min(num_rounds, max_smps) + chan_dim - 1) / chan_dim;
    uint32_t num_blocks = round_stride * chan_dim;
    conv_batch_stats_mean_fwd_atomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, round_stride, spatial_dim, chan_dim, batch_sz, x, mean);
  }
}

__global__ void conv_batch_stats_var_fwd_nonatomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    float *var)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x;
  float value = 0.0f;
  if (chan_idx < chan_dim) {
    float m = mean[chan_idx];
    for (uint32_t round = 0; round < num_rounds; round++) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        float residual = x[idx] - m;
        value += residual * residual;
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    var[chan_idx] += cache[0] / ((float)(spatial_dim * (batch_sz - 1)));
  }
}

__global__ void conv_batch_stats_var_fwd_atomic_f32_kernel(
    uint32_t num_rounds,
    uint32_t round_stride,
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    float *var)
{
  __shared__ float cache[1024+32];
  uint32_t chan_idx = blockIdx.x % chan_dim;
  uint32_t round_start = blockIdx.x / chan_dim;
  float value = 0.0f;
  if (chan_idx < chan_dim && round_start < round_stride) {
    float m = mean[chan_idx];
    for (uint32_t round = round_start; round < num_rounds; round += round_stride) {
      uint32_t round_offset = round * blockDim.x;
      uint32_t round_idx = round_offset + threadIdx.x;
      uint32_t spatial_idx = round_idx % spatial_dim;
      uint32_t batch_idx = round_idx / spatial_dim;
      if (spatial_idx < spatial_dim && batch_idx < batch_sz) {
        uint32_t idx = spatial_idx + spatial_dim * (chan_idx + chan_dim * batch_idx);
        float residual = x[idx] - m;
        value += residual * residual;
      }
    }
  }
  cache[OFFSET_BANK(threadIdx.x)] = value;
  __syncthreads();
  for (uint32_t s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x & (2 * s - 1)) == 0 && (threadIdx.x + s) < blockDim.x) {
      cache[OFFSET_BANK(threadIdx.x)] += cache[OFFSET_BANK(threadIdx.x + s)];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&var[chan_idx], cache[0] / ((float)(spatial_dim * (batch_sz - 1))));
  }
}

extern "C" void arraydiff_cuda_kernel_conv_batch_stats_var_fwd_nonatomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    float *var,
    cudaStream_t stream)
{
  // XXX: `var` should be zeroed.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  uint32_t num_blocks = chan_dim;
  conv_batch_stats_var_fwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
      num_rounds, spatial_dim, chan_dim, batch_sz, x, mean, var);
}

extern "C" void arraydiff_cuda_kernel_conv_batch_stats_var_fwd_atomic_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    float *var,
    cudaStream_t stream)
{
  // XXX: `var` should be zeroed.
  const uint32_t max_smps = 24; // FIXME: get from cudaDeviceProp.
  uint32_t num_rounds = (spatial_dim * batch_sz + 1024-1) / 1024;
  if (chan_dim >= min(num_rounds, max_smps)) {
    uint32_t num_blocks = chan_dim;
    conv_batch_stats_var_fwd_nonatomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, spatial_dim, chan_dim, batch_sz, x, mean, var);
  } else {
    uint32_t round_stride = (min(num_rounds, max_smps) + chan_dim - 1) / chan_dim;
    uint32_t num_blocks = round_stride * chan_dim;
    conv_batch_stats_var_fwd_atomic_f32_kernel<<<num_blocks, 1024, 0, stream>>>(
        num_rounds, round_stride, spatial_dim, chan_dim, batch_sz, x, mean, var);
  }
}

__global__ void conv_batch_stats_bwd_f32_kernel(
    uint32_t spatial_dim,
    uint32_t chan_dim,
    uint32_t batch_sz,
    const float *x,
    const float *mean,
    const float *mean_grad,
    const float *var_grad,
    float *x_grad)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t spatial_idx = idx % spatial_dim;
  uint32_t chan_idx = (idx / spatial_dim) % chan_dim;
  uint32_t batch_idx = (idx / spatial_dim) / chan_dim;
  if (spatial_idx < spatial_dim && chan_idx < chan_dim && batch_idx < batch_sz) {
    x_grad[idx] += mean_grad[chan_idx] / ((float)(spatial_dim * batch_sz)) + 2.0f * var_grad[chan_idx] * (x[idx] - mean[chan_idx]) / ((float)(spatial_dim * (batch_sz - 1)));;
  }
}

extern "C" void arraydiff_cuda_kernel_conv_batch_stats_bwd_f32(
    size_t spatial_dim,
    size_t chan_dim,
    size_t batch_sz,
    const float *x,
    const float *mean,
    const float *mean_grad,
    const float *var_grad,
    float *x_grad,
    cudaStream_t stream)
{
  uint32_t n = spatial_dim * chan_dim * batch_sz;
  conv_batch_stats_bwd_f32_kernel<<<(n+1024-1)/1024, 1024, 0, stream>>>(
      spatial_dim, chan_dim, batch_sz, x, mean, mean_grad, var_grad, x_grad);
}
