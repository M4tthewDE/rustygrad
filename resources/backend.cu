#include <float.h>
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

__global__ void sqrt_kernel(double *a, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = sqrtf(a[i]);
  }
}

extern "C" void rusty_sqrt(double *a, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  sqrt_kernel<<<gridDim, blockDim>>>(a, c, n);
}

__global__ void log_kernel(double *a, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = log2f(a[i]);
  }
}

extern "C" void rusty_log(double *a, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  log_kernel<<<gridDim, blockDim>>>(a, c, n);
}

__global__ void relu_kernel(double *a, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (a[i] < 0.0) {
      c[i] = 0.0;
    } else {
      c[i] = a[i];
    }
  }
}

extern "C" void relu(double *a, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  relu_kernel<<<gridDim, blockDim>>>(a, c, n);
}

__global__ void sigmoid_kernel(double *a, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = (1.0 / (1.0 + expf(-a[i])));
  }
}

extern "C" void sigmoid(double *a, double *c, int n) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  sigmoid_kernel<<<gridDim, blockDim>>>(a, c, n);
}

__global__ void max_kernel(double *a, double *max, int n) {
  extern __shared__ int shared[];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  double localMax = -DBL_MAX;

  // Local reduction
  for (int i = index; i < n; i += stride) {
    localMax = fmaxf(localMax, a[i]);
  }

  // Store local max in shared memory
  shared[threadIdx.x] = localMax;
  __syncthreads();

  // Reduction within a block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
    }
    __syncthreads();
  }

  // First thread in each block writes the result
  if (threadIdx.x == 0) {
    max[blockIdx.x] = shared[0];
  }
}

extern "C" void rusty_max(double *a, double *c, int n) {
  const int blockSize = 256;
  const int numBlocks = (n + blockSize - 1) / blockSize;
  dim3 blockDim(blockSize);
  dim3 gridDim(numBlocks);

  double *max;
  cudaMalloc(&max, numBlocks);
  double *result_max;
  cudaMalloc(&result_max, numBlocks);

  max_kernel<<<gridDim, blockDim>>>(a, max, n);

  // Copy results back to host
  cudaMemcpy(result_max, max, numBlocks * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Final reduction on host
  double finalMax = -DBL_MAX;
  for (int i = 0; i < numBlocks; ++i) {
    finalMax = fmaxf(finalMax, result_max[i]);
  }

  cudaMemcpy(c, &finalMax, 1, cudaMemcpyHostToDevice);
}
