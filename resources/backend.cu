#include <float.h>
#include <stdio.h>

// Reading material:
//
// http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
// https://docs.nvidia.com/deeplearning/performance/pdf/GPU-Performance-Background-User-Guide.pdf
// https://developer.nvidia.com/nvidia-visual-profiler

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
    c[i] = fmaxf(a[i], 0.0);
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
  extern __shared__ double shared[];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  double localMax = -DBL_MAX;

  // Local reduction
  for (int i = index; i < n; i += stride) {
    localMax = fmax(localMax, a[i]);
  }

  // Store local max in shared memory
  shared[threadIdx.x] = localMax;
  __syncthreads();

  // Reduction within a block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + s]);
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
  cudaMalloc(&max, numBlocks * sizeof(double));

  max_kernel<<<gridDim, blockDim, blockSize * sizeof(double)>>>(a, max, n);

  double *result_max = new double[numBlocks];
  cudaMemcpy(result_max, max, numBlocks * sizeof(double),
             cudaMemcpyDeviceToHost);

  double finalMax = -DBL_MAX;
  for (int i = 0; i < numBlocks; ++i) {
    finalMax = fmax(finalMax, result_max[i]);
  }

  // copying it back to device to be used by other ops
  // FIXME: is it possible to calculate finalMax on device?
  cudaMemcpy(c, &finalMax, 1 * sizeof(double), cudaMemcpyHostToDevice);
}

__global__ void min_kernel(double *a, double *max, int n) {
  extern __shared__ double shared[];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  double localMin = DBL_MAX;

  // Local reduction
  for (int i = index; i < n; i += stride) {
    localMin = fmin(localMin, a[i]);
  }

  // Store local max in shared memory
  shared[threadIdx.x] = localMin;
  __syncthreads();

  // Reduction within a block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] = fmin(shared[threadIdx.x], shared[threadIdx.x + s]);
    }
    __syncthreads();
  }

  // First thread in each block writes the result
  if (threadIdx.x == 0) {
    max[blockIdx.x] = shared[0];
  }
}

extern "C" void rusty_min(double *a, double *c, int n) {
  const int blockSize = 256;
  const int numBlocks = (n + blockSize - 1) / blockSize;
  dim3 blockDim(blockSize);
  dim3 gridDim(numBlocks);

  double *min;
  cudaMalloc(&min, numBlocks * sizeof(double));

  min_kernel<<<gridDim, blockDim, blockSize * sizeof(double)>>>(a, min, n);

  double *result_min = new double[numBlocks];
  cudaMemcpy(result_min, min, numBlocks * sizeof(double),
             cudaMemcpyDeviceToHost);

  double finalMin = DBL_MAX;
  for (int i = 0; i < numBlocks; ++i) {
    finalMin = fmin(finalMin, result_min[i]);
  }

  cudaMemcpy(c, &finalMin, 1 * sizeof(double), cudaMemcpyHostToDevice);
}

// A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel(double *A, double *B, double *C, int M, int K,
                              int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

extern "C" void matmul(double *a, double *b, double *c, int M, int K, int N) {
  int blockSize = 16;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize,
               1);
  matmul_kernel<<<dimGrid, dimBlock>>>(a, b, c, M, K, N);
}

__global__ void expand_kernel(double *input, double *output, int output_length,
                              int dim_count, size_t *old_shape,
                              size_t *new_shape) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < output_length) {
    size_t idx = 0;
    size_t factor = 1;
    size_t index = i;

    for (int k = dim_count - 1; k >= 0; k--) {
      size_t size_new = new_shape[k];
      size_t size_old = old_shape[k];
      int old_index = 0;

      if (size_old != 1) {
        old_index = i % size_new;
      }

      idx += old_index * factor;
      factor *= size_old;
      i /= size_new;
    }

    output[index] = input[idx];
  }
}

extern "C" void expand(double *input, double *output, int output_length,
                       int dim_count, size_t *old_shape, size_t *new_shape) {
  dim3 blockDim(256);
  dim3 gridDim((output_length + blockDim.x - 1) / blockDim.x);

  expand_kernel<<<gridDim, blockDim>>>(input, output, output_length, dim_count,
                                       old_shape, new_shape);
}

__global__ void pad2d_kernel(double *input, double *output, int input_length,
                             int dim_count, size_t *shape, size_t *new_shape,
                             size_t *padding) {
  extern __shared__ int shared_mem[];
  int *multi_dim_index = shared_mem + threadIdx.x * dim_count;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < input_length) {
    size_t temp_index = i;

    for (int k = dim_count - 1; k >= 0; k--) {
      size_t size = shape[k];
      multi_dim_index[k] = temp_index % size;
      temp_index /= size;
    }

    // bottom and right padding is added in the initialization
    if (dim_count >= 2) {
      multi_dim_index[dim_count - 2] += padding[2]; // top padding
      multi_dim_index[dim_count - 1] += padding[0]; // left padding
    }

    size_t new_index = 0;
    size_t stride = 1;
    for (int k = dim_count - 1; k >= 0; k--) {
      size_t size = new_shape[k];
      size_t index = multi_dim_index[k];

      new_index += index * stride;
      stride *= size;
    }

    output[new_index] = input[i];
  }
}

extern "C" void pad2d(double *input, double *output, int input_length,
                      int dim_count, size_t *shape, size_t *new_shape,
                      size_t *padding) {
  dim3 blockDim(256);
  dim3 gridDim((input_length + blockDim.x - 1) / blockDim.x);

  size_t sharedMemSize = blockDim.x * dim_count * sizeof(int);

  pad2d_kernel<<<gridDim, blockDim, sharedMemSize>>>(
      input, output, input_length, dim_count, shape, new_shape, padding);
}

__global__ void permute_kernel(double *input, double *output, int input_length,
                               int dim_count, size_t *shape, size_t *new_shape,
                               size_t *dims) {
  extern __shared__ int shared_mem[];
  int *multi_dim_index = shared_mem + threadIdx.x * dim_count;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < input_length) {
    size_t temp_index = i;

    for (int k = dim_count - 1; k >= 0; k--) {
      size_t size = shape[k];
      multi_dim_index[k] = temp_index % size;
      temp_index /= size;
    }

    size_t new_index = 0;
    size_t stride = 1;
    for (int k = dim_count - 1; k >= 0; k--) {
      size_t size = new_shape[k];
      size_t index = multi_dim_index[dims[k]];

      new_index += index * stride;
      stride *= size;
    }

    output[new_index] = input[i];
  }
}

extern "C" void permute(double *input, double *output, int input_length,
                        int dim_count, size_t *shape, size_t *new_shape,
                        size_t *dims) {
  dim3 blockDim(256);
  dim3 gridDim((input_length + blockDim.x - 1) / blockDim.x);

  size_t sharedMemSize = blockDim.x * dim_count * sizeof(int);

  permute_kernel<<<gridDim, blockDim, sharedMemSize>>>(
      input, output, input_length, dim_count, shape, new_shape, dims);
}

__global__ void sum_kernel(double *input, int n, size_t *input_shape,
                           size_t input_shape_len, size_t *dims,
                           size_t dims_len, size_t *reduced_shape,
                           size_t reduced_shape_len, double *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    int offset = 0;
    int reduced_shape_idx = 0;
    int new_index = 0;
    for (int j = 0; j < input_shape_len; j++) {
      int count = 1;
      for (int k = 0; k <= j; k++) {
        count *= input_shape[k];
      }
      size_t index = (i - offset) / (n / count);

      bool in_dims = false;
      for (int k = 0; k < dims_len; k++) {
        if (dims[k] == j) {
          in_dims = true;
          break;
        }
      }

      if (!in_dims) {
        if (reduced_shape_idx == reduced_shape_len - 1) {
          new_index += index;
        } else {
          new_index +=
              index * reduced_shape[reduced_shape_len - reduced_shape_idx - 1];
        }
        reduced_shape_idx++;
      }
      offset += (n / count) * index;
    }

    atomicAdd(&result[new_index], input[i]);
  }
}

extern "C" void sum(double *input, double *result, size_t n,
                    size_t *input_shape, size_t input_shape_len, size_t *dims,
                    size_t dims_len, size_t *reduced_shape,
                    size_t reduced_shape_len) {
  dim3 blockDim(256);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

  sum_kernel<<<gridDim, blockDim>>>(input, n, input_shape, input_shape_len,
                                    dims, dims_len, reduced_shape,
                                    reduced_shape_len, result);
}

__global__ void sum_pool2d_kernel(double *input, double *result,
                                  size_t *input_shape, size_t *result_shape,
                                  size_t *kernel, double init_val,
                                  size_t stride) {
  size_t batch = input_shape[0];
  size_t channels = input_shape[1];
  size_t height = input_shape[2];
  size_t width = input_shape[3];

  size_t n = blockIdx.z; // One block per batch element
  size_t c = blockIdx.y; // One block per channel
  size_t i =
      blockIdx.x * blockDim.x + threadIdx.x; // Cover the width of the output
  size_t j =
      blockIdx.x * blockDim.y + threadIdx.y; // Cover the height of the output

  if (n >= batch || c >= channels || i >= result_shape[2] ||
      j >= result_shape[3])
    return;

  double result_val = init_val;
  for (int ki = 0; ki < kernel[0]; ki++) {
    for (int kj = 0; kj < kernel[1]; kj++) {
      size_t row = i * stride + ki;
      size_t col = j * stride + kj;
      if (row < height && col < width) { // Check boundaries
        size_t idx = n * (channels * height * width) + c * (height * width) +
                     row * width + col;
        result_val += input[idx];
      }
    }
  }

  size_t result_idx = n * (channels * result_shape[2] * result_shape[3]) +
                      c * (result_shape[2] * result_shape[3]) +
                      i * result_shape[3] + j;
  result[result_idx] = result_val;
}

extern "C" void sum_pool2d(double *input, double *result, size_t *input_shape,
                           size_t *result_shape, size_t *kernel,
                           double init_val, size_t stride, size_t batch,
                           size_t channels, size_t output_height,
                           size_t output_width) {
  // Calculate grid and block sizes
  dim3 threadsPerBlock(
      16,
      16); // 16x16 threads per block is a common choice, adjust as necessary
  dim3 numBlocks((output_height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (output_width + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 batch * channels); // One block per element in the batch and
                                    // channel dimensions

  sum_pool2d_kernel<<<numBlocks, threadsPerBlock>>>(
      input, result, input_shape, result_shape, kernel, init_val, stride);
}

// FIXME: parallelize
__global__ void max_pool2d_kernel(double *input, double *result,
                                  size_t *input_shape, size_t *result_shape,
                                  size_t *kernel, double init_val,
                                  size_t stride) {
  size_t batch = input_shape[0];
  size_t channels = input_shape[1];
  size_t height = input_shape[2];
  size_t width = input_shape[3];

  size_t result_idx = 0;
  for (int n = 0; n < batch; n++) {
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < result_shape[2]; i++) {
        for (int j = 0; j < result_shape[3]; j++) {
          double result_val = init_val;
          for (int ki = 0; ki < kernel[0]; ki++) {
            for (int kj = 0; kj < kernel[1]; kj++) {
              size_t row = i * stride + ki;
              size_t col = j * stride + kj;
              size_t idx = n * (channels * height * width) +
                           c * (height * width) + row * width + col;

              result_val = max(result_val, input[idx]);
            }
          }
          result[result_idx] = result_val;
          result_idx++;
        }
      }
    }
  }
}

extern "C" void max_pool2d(double *input, double *result, size_t *input_shape,
                           size_t *result_shape, size_t *kernel,
                           double init_val, size_t stride) {
  max_pool2d_kernel<<<1, 1>>>(input, result, input_shape, result_shape, kernel,
                              init_val, stride);
}

__device__ size_t index_4d_to_1d(size_t *shape, size_t n, size_t c, size_t h,
                                 size_t w) {
  size_t height = shape[2];
  size_t width = shape[3];
  size_t channels = shape[1];
  return n * (channels * height * width) + c * (height * width) + h * width + w;
}

__global__ void conv2d_kernel(double *input, double *result,
                              size_t *input_shape, size_t *result_shape,
                              size_t groups, size_t *kernel_shape,
                              double *kernel, size_t *strides) {
  size_t n = input_shape[0];
  size_t c_out = kernel_shape[0];
  size_t output_height = ((input_shape[2] - kernel_shape[2]) / strides[0]) + 1;
  size_t output_width = ((input_shape[3] - kernel_shape[3]) / strides[1]) + 1;

  // Calculate indices based on thread and block IDs
  size_t n_index = blockIdx.x;
  size_t c_out_index = blockIdx.y;
  size_t i = blockIdx.z / output_width;
  size_t j = blockIdx.z % output_width;

  if (n_index >= n || c_out_index >= c_out || i >= output_height ||
      j >= output_width)
    return;

  size_t c_in = input_shape[1];
  size_t height = input_shape[2];
  size_t width = input_shape[3];

  size_t kernel_height = kernel_shape[2];
  size_t kernel_width = kernel_shape[3];

  size_t c_in_per_group = c_in / groups;
  size_t group = c_out_index / (c_out / groups);

  double value = 0.0;
  for (int c_in_index = group * c_in_per_group;
       c_in_index < (group + 1) * c_in_per_group; c_in_index++) {
    for (int k_row = 0; k_row < kernel_height; k_row++) {
      for (int k_col = 0; k_col < kernel_width; k_col++) {
        size_t row = i * strides[0] + k_row;
        size_t col = j * strides[1] + k_col;
        if (row < height && col < width) {
          value +=
              input[index_4d_to_1d(input_shape, n_index, c_in_index, row,
                                   col)] *
              kernel[index_4d_to_1d(kernel_shape, c_out_index,
                                    c_in_index % c_in_per_group, k_row, k_col)];
        }
      }
    }
  }

  size_t result_idx = index_4d_to_1d(result_shape, n_index, c_out_index, i, j);
  result[result_idx] = value;
}

extern "C" void conv2d(double *input, double *result, size_t *input_shape,
                       size_t *result_shape, size_t groups,
                       size_t *kernel_shape, double *kernel, size_t *strides,
                       size_t n, size_t c_out, size_t total_output_elements) {
  dim3 blockDim(1, 1, 1);
  dim3 gridDim(n, c_out, total_output_elements);
  conv2d_kernel<<<gridDim, blockDim>>>(input, result, input_shape, result_shape,
                                       groups, kernel_shape, kernel, strides);
}
