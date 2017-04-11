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

#[cfg(feature = "cuda")] use cuda::ffi::runtime::{cudaStream_t};
//use libc::*;

#[link(name = "arraydiff_kernels", kind = "static")]
extern "C" {
  // Special map functions.
  pub fn arraydiff_kernel_rect_fwd_f32(dim: usize, x: *const f32, y: *mut f32);
  pub fn arraydiff_kernel_rect_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32);
  pub fn arraydiff_kernel_logistic_fwd_f32(dim: usize, x: *const f32, y: *mut f32);
  pub fn arraydiff_kernel_logistic_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32);
  pub fn arraydiff_kernel_logistic_rbwd_f32(dim: usize, x: *const f32, r_x: *const f32, dy: *const f32, r_dy: *const f32, r_dx: *mut f32);
  pub fn arraydiff_kernel_logistic_bwd2_f32(dim: usize, x: *const f32, dy: *const f32, dy2: *const f32, dx2: *mut f32);
  pub fn arraydiff_kernel_tanh_fwd_f32(dim: usize, x: *const f32, y: *mut f32);
  pub fn arraydiff_kernel_tanh_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32);
  pub fn arraydiff_kernel_tanh_rbwd_f32(dim: usize, x: *const f32, r_x: *const f32, dy: *const f32, r_dy: *const f32, r_dx: *mut f32);
}

#[cfg(feature = "cuda")]
#[link(name = "arraydiff_cuda_kernels", kind = "static")]
extern "C" {
  // Special map functions.
  pub fn arraydiff_cuda_kernel_rect_fwd_f32(dim: usize, x: *const f32, y: *mut f32, stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_rect_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32, stream: cudaStream_t);

  pub fn arraydiff_cuda_kernel_cast_u8_to_f32(dim: usize, x: *const u8, y: *mut f32, stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_cast_u8x4_to_f32x4(dim: usize, x: *const u8, y: *mut f32, stream: cudaStream_t);

  pub fn arraydiff_cuda_kernel_conv_bcast_mult_add_fwd_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      scale: *const f32,
      shift: *const f32,
      y: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_bcast_mult_add_param_bwd_nonatomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      scale: *const f32,
      shift: *const f32,
      y_grad: *const f32,
      scale_grad: *mut f32,
      shift_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_bcast_mult_add_param_bwd_atomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      scale: *const f32,
      shift: *const f32,
      y_grad: *const f32,
      scale_grad: *mut f32,
      shift_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_bcast_mult_add_input_bwd_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      scale: *const f32,
      y_grad: *const f32,
      x_grad: *mut f32,
      stream: cudaStream_t);

  pub fn arraydiff_cuda_kernel_conv_normalize_fwd_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *const f32,
      epsilon: f32,
      y: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_normalize_var_bwd_nonatomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *const f32,
      y_grad: *const f32,
      epsilon: f32,
      var_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_normalize_var_bwd_atomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *const f32,
      y_grad: *const f32,
      epsilon: f32,
      var_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_normalize_mean_bwd_nonatomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *const f32,
      var_grad: *const f32,
      y_grad: *const f32,
      epsilon: f32,
      mean_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_normalize_mean_bwd_atomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *const f32,
      var_grad: *const f32,
      y_grad: *const f32,
      epsilon: f32,
      mean_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_normalize_input_bwd_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      var: *const f32,
      y_grad: *const f32,
      epsilon: f32,
      x_grad: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_batch_stats_mean_fwd_nonatomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_batch_stats_mean_fwd_atomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_batch_stats_var_fwd_nonatomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_batch_stats_var_fwd_atomic_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      var: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_conv_batch_stats_bwd_f32(
      spatial_dim: usize,
      chan_dim: usize,
      batch_sz: usize,
      x: *const f32,
      mean: *const f32,
      mean_grad: *const f32,
      var_grad: *const f32,
      x_grad: *mut f32,
      stream: cudaStream_t);

  pub fn arraydiff_cuda_kernel_max_pool_fwd_f32(
      x_w: usize, x_h: usize, chan_dim: usize, batch_sz: usize,
      y_w: usize, y_h: usize,
      kernel_w: usize, kernel_h: usize,
      stride_w: usize, stride_h: usize,
      pad_w: usize, pad_h: usize,
      x: *const f32,
      maybe_y: *mut f32,
      maybe_mask: *mut i32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_max_pool_bwd_f32(
      x_w: usize, x_h: usize, chan_dim: usize, batch_sz: usize,
      y_w: usize, y_h: usize,
      kernel_w: usize, kernel_h: usize,
      stride_w: usize, stride_h: usize,
      pad_w: usize, pad_h: usize,
      dy: *const f32,
      mask: *const i32,
      dx: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_avg_pool_fwd_f32(
      x_w: usize, x_h: usize, chan_dim: usize, batch_sz: usize,
      y_w: usize, y_h: usize,
      kernel_w: usize, kernel_h: usize,
      stride_w: usize, stride_h: usize,
      pad_w: usize, pad_h: usize,
      x: *const f32,
      y: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_avg_pool_bwd_f32(
      x_w: usize, x_h: usize, chan_dim: usize, batch_sz: usize,
      y_w: usize, y_h: usize,
      kernel_w: usize, kernel_h: usize,
      stride_w: usize, stride_h: usize,
      pad_w: usize, pad_h: usize,
      dy: *const f32,
      dx: *mut f32,
      stream: cudaStream_t);

  pub fn arraydiff_cuda_kernel_blockreduce_max_argmax_f32(
      block_dim: usize,
      num_blocks: usize,
      x: *const f32,
      x_max: *mut f32,
      x_argmax: *mut u32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_blockreduce_sum_f32(
      block_dim: usize,
      num_blocks: usize,
      x: *const f32,
      x_sum: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_reduce_one_hot_fwd_f32(
      dim: usize,
      batch_sz: usize,
      x: *const f32,
      index: *const u32,
      y: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_reduce_one_hot_bwd_f32(
      dim: usize,
      batch_sz: usize,
      dy: *const f32,
      index: *const u32,
      dx: *mut f32,
      stream: cudaStream_t);

  pub fn arraydiff_cuda_kernel_block_softmax_fwd_f32(
      block_dim: usize,
      num_blocks: usize,
      x: *const f32,
      y: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_softmax_nll_loss_fwd_f32(
      dim: usize,
      batch_sz: usize,
      y: *const f32,
      t: *const u32,
      loss: *mut f32,
      stream: cudaStream_t);
  pub fn arraydiff_cuda_kernel_softmax_nll_loss_bwd_f32(
      dim: usize,
      batch_sz: usize,
      y: *const f32,
      t: *const u32,
      df: *const f32,
      dx: *mut f32,
      stream: cudaStream_t);
}
