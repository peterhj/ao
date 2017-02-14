use libc::*;

#[link(name = "arraydiff_kernels", kind = "static")]
extern "C" {
  pub fn arraydiff_kernel_rect_fwd_f32(dim: usize, x: *const f32, y: *mut f32);
  pub fn arraydiff_kernel_rect_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32);
  pub fn arraydiff_kernel_logistic_fwd_f32(dim: usize, x: *const f32, y: *mut f32);
  pub fn arraydiff_kernel_logistic_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32);
  pub fn arraydiff_kernel_tanh_fwd_f32(dim: usize, x: *const f32, y: *mut f32);
  pub fn arraydiff_kernel_tanh_bwd_f32(dim: usize, x: *const f32, dy: *const f32, dx: *mut f32);
}
