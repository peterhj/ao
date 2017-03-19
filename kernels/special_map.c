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
    float y_i = 1.0f / (1.0f + expf(-x_i));
    y[i] = y_i;
  }
}

void arraydiff_kernel_logistic_bwd_f32(size_t dim, const float *x, const float *dy, float *dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float y_i = 1.0f / (1.0f + expf(-x_i));
    dx[i] += y_i * (1.0f - y_i);
  }
}

void arraydiff_kernel_logistic_rbwd_f32(size_t dim, const float *x, const float *r_x, const float *dy, const float *r_dy, float *r_dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float y_i = 1.0f / (1.0f + expf(-x_i));
    /*r_dx[i] += r_dy[i] * (y_i * (1.0f - y_i)) + dy[i] * r_x[i] * (y_i * (1.0f - y_i) * (1.0f - 2.0f * y_i));*/
    r_dx[i] += (r_dy[i] + dy[i] * r_x[i] * (1.0f - 2.0f * y_i)) * (y_i * (1.0f - y_i));
  }
}

void arraydiff_kernel_logistic_bwd2_f32(size_t dim, const float *x, const float *dy, const float *dy2, float *dx2) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float y_i = 1.0f / (1.0f + expf(-x_i));
    /*dx2[i] += dy2[i] * (y_i * (1.0f - y_i)) * (y_i * (1.0f - y_i)) + dy[i] * (y_i * (1.0f - y_i) * (1.0f - 2.0f * y_i));*/
    dx2[i] += (dy2[i] * (y_i * (1.0f - y_i)) + dy[i] * (1.0f - 2.0f * y_i)) * (y_i * (1.0f - y_i));
  }
}

void arraydiff_kernel_tanh_fwd_f32(size_t dim, const float *x, float *y) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float hi = expf(x_i);
    float lo = expf(-x_i);
    float t = (hi - lo) / (hi + lo);
    y[i] = t;
  }
}

void arraydiff_kernel_tanh_bwd_f32(size_t dim, const float *x, const float *dy, float *dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float hi = expf(x_i);
    float lo = expf(-x_i);
    float s = 2.0f / (hi + lo);
    dx[i] += dy[i] * (s * s);
  }
}

void arraydiff_kernel_tanh_rbwd_f32(size_t dim, const float *x, const float *r_x, const float *dy, const float *r_dy, float *r_dx) {
  for (size_t i = 0; i < dim; i++) {
    float x_i = x[i];
    float hi = expf(x_i);
    float lo = expf(-x_i);
    float t = (hi - lo) / (hi + lo);
    float s = 2.0f / (hi + lo);
    /*r_dx[i] += r_dy[i] * (s * s) + dy[i] * r_x[i] * (-2.0f * t * s * s);*/
    r_dx[i] += (r_dy[i] + dy[i] * r_x[i] * (-2.0f * t)) * (s * s);
  }
}
