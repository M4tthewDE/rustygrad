#include <stdio.h>

__global__ void add_kernel(double *a, double *b, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

extern "C" void add(double *a, double *b, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  add_kernel<<<gridDim, blockDim>>>(a, b, c, n);
}

__global__ void sub_kernel(double *a, double *b, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] - b[i];
  }
}

extern "C" void sub(double *a, double *b, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  sub_kernel<<<gridDim, blockDim>>>(a, b, c, n);
}

__global__ void mul_kernel(double *a, double *b, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * b[i];
  }
}

extern "C" void mul(double *a, double *b, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  mul_kernel<<<gridDim, blockDim>>>(a, b, c, n);
}

__global__ void div_kernel(double *a, double *b, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] / b[i];
  }
}

// not named "div" because it exists already
extern "C" void division(double *a, double *b, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  div_kernel<<<gridDim, blockDim>>>(a, b, c, n);
}
