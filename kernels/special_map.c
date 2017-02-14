#include <math.h>
#include <stdlib.h>

void arraydiff_kernel_rect_fwd_f32(size_t dim, const float *x, float *y) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    y[i] = x_i * (x_i > 0.0f);
  }
}

void arraydiff_kernel_rect_bwd_f32(size_t dim, const float *x, const float *dy, float *dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    dx[i] += dy[i] * (x_i > 0.0f);
  }
}

void arraydiff_kernel_logistic_fwd_f32(size_t dim, const float *x, float *y) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    y[i] = 1.0f / (1.0f + expf(-x_i));
  }
}

void arraydiff_kernel_logistic_bwd_f32(size_t dim, const float *x, const float *dy, float *dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float y_i = 1.0f / (1.0f + expf(-x_i));
    dx[i] += y_i * (1.0f - y_i);
  }
}

void arraydiff_kernel_tanh_fwd_f32(size_t dim, const float *x, float *y) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float hi = expf(x_i);
    float lo = expf(-x_i);
    y[i] = (hi - lo) / (hi + lo);
  }
}

void arraydiff_kernel_tanh_bwd_f32(size_t dim, const float *x, const float *dy, float *dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float hi = expf(x_i);
    float lo = expf(-x_i);
    float s = 2.0f / (hi + lo);
    dx[i] += dy[i] * s * s;
  }
}

/*void arraydiff_kernel_tanh_rbwd_f32(size_t dim, const float *x, const float *dy, float *r_dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float hi = expf(x_i);
    float lo = expf(-x_i);
    //y[i] = (hi - lo) / (hi + lo);
    float s = 2.0f / (hi + lo);
    dx[i] += dy[i] * s * s;
  }
}*/
