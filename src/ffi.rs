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
