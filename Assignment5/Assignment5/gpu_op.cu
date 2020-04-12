#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here (DONE)*/
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void array_set_kernel(int n, float *arr_data, float value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    arr_data[x] = value;
}

__global__ void broadcast_kernel(int in, int out, const float *in_data,
                                                        float *out_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= in) {
        return;
    }
    for (int i = x; i < out; i += in) {
        out_data[i] = in_data[x];
    }
}

__global__ void reduce_sum_axis_zero_kernel(int in, int out, const
                                            float *in_data, float *out_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= out) {
        return;
    }
    out_data[x] = 0;
    for (int i = x; i < in; i += out) {
        out_data[x] += in_data[i];
    }
}

__global__ void matrix_add_kernel(int n, const float *input_data_A, const
                                  float *input_data_B, float *output_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    output_data[x] = input_data_A[x] + input_data_B[x];
}

__global__ void matrix_add_by_const_kernel(int n, const float *input_data,
                                          float value, float *output_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    output_data[x] = input_data[x] + value;
}

__global__ void matrix_mul_kernel(int n, const float *input_data_A, const
                                  float *input_data_B, float *output_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    output_data[x] = input_data_A[x] * input_data_B[x];
}

__global__ void matrix_mul_by_const_kernel(int n, const float *input_data,
                                           float value, float *output_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    output_data[x] = input_data[x] * value;
}

__global__ void MatMulKernel(const float *A, const float *B, float *C,
                             int rowA, int rowB, int colA, int colB,
                             bool transA, bool transB) {

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue = 0.0;

    if(!transA && !transB){
        if(r >= rowA || c >= colB) return;

        for (int i = 0; i < colA; ++i)
            Cvalue += (A[r * colA + i]) * (B[i * colB + c]);
        C[r * colB + c] = Cvalue;
    }
    else if(transA && transB){
        if(r >= colA || c >= rowB) return;

        for (int i = 0; i < rowA; ++i)
            Cvalue += (A[i * colA + r]) * (B[c * colB + i]);
        C[r * rowB + c] = Cvalue;
    }
    else if(transA && !transB){
        if(r >= colA || c >= colB) return;

        for (int i = 0; i < rowA; ++i)
            Cvalue += (A[i * colA + r]) * (B[i * colB + c]);
        C[r * colB + c] = Cvalue;
    }
    else if(!transA && transB){
        if(r >= rowA || c >= rowB) return;

        for (int i = 0; i < colA; ++i)
            Cvalue += (A[r * colA + i]) * (B[c * colB + i]);
        C[r * rowB + c] = Cvalue;
    }
}

__global__ void relu_kernel(int n, const float *input_data, float *output_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    if (input_data[x] > 0.0) {
        output_data[x] = input_data[x];
    }
    else {
        output_data[x] = 0.0;
    }
}

__global__ void relu_gradient_kernel(int n, const float *input_data, const
                                     float *input_grad_data, float *output_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) {
        return;
    }
    if (input_data[x] > 0.0) {
        output_data[x] = input_grad_data[x];
    }
    else {
        output_data[x] = 0.0;
    }
}

__global__ void matrix_softmax_kernel(int nrow, int ncol, const
                                      float *input_data, float *output_data) {
    int x = threadIdx.y * blockDim.x + threadIdx.x;
    if (x >= nrow) {
        return;
    }
    input_data += x * ncol;
    output_data += x * ncol;
    float maxval = *input_data;
    float sum = 0;

    // Find max for a row.
    for (int i = 1; i < ncol; ++i) {
        maxval = max(maxval, input_data[i]);
    }

    // Deduct by max for a row, and raise to exp.
    for (int i = 0; i < ncol; ++i) {
        sum += exp(input_data[i] - maxval);
    }

    for (int i = 0; i < ncol; ++i) {
        output_data[i] = exp(input_data[i] - maxval) / sum;
    }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here (Done)*/
  int n = 1;

  dim3 blocks;
  dim3 threads;

  float *arr_data = (float *)arr->data;

  for(int i = 0; i < arr->ndim; i++){
    n *= arr->shape[i];
  }

  if(n <= 1024){
    blocks.x = 1;
    threads.x = n;
  } else {
    blocks.x = (n + 1023) / 1024;
    threads.x = 1024;
  }

  array_set_kernel<<<blocks, threads>>>(n, arr_data, value);

  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int in = 1;
  int out = 1;

  dim3 blocks;
  dim3 threads;

  float *out_data = (float *)output->data;
  const float *in_data = (const float *)input->data;

  for (int i = 0; i < input->ndim; i++) {
      in *= input->shape[i];
  }

  for (int i = 0; i < output->ndim; i++) {
      out *= output->shape[i];
  }

  if (in <= 1024) {
      blocks.x = 1;
      threads.x = in;
  } else {
      blocks.x = (in + 1023) / 1024;
      threads.x = 1024;
  }

  broadcast_kernel<<<blocks, threads>>>(in, out, in_data, out_data);

  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int in = 1;
  int out = 1;

  dim3 blocks;
  dim3 threads;

  float *out_data = (float *)output->data;
  const float *in_data = (const float *)input->data;

  for (int i = 0; i < input->ndim; i++) {
      in *= input->shape[i];
  }
  for (int i = 0; i < output->ndim; i++) {
      out *= output->shape[i];
  }

  if (out <= 1024) {
      blocks.x = 1;
      threads.x = out;
  } else {
      blocks.x = (out + 1023) / 1024;
      threads.x = 1024;
  }

  reduce_sum_axis_zero_kernel<<<blocks, threads>>>(in, out, in_data, out_data);

  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int n = 1;

  dim3 blocks;
  dim3 threads;

  const float *input_data_A = (const float *)matA->data;
  const float *input_data_B = (const float *)matB->data;

  float *output_data = (float *)output->data;

  for (int i = 0; i < matA->ndim; i++) {
      n *= matA->shape[i];
  }

  if (n <= 1024) {
      blocks.x = 1;
      threads.x = n;
  } else {
      blocks.x = (n + 1023) / 1024;
      threads.x = 1024;
  }

  matrix_add_kernel<<<blocks, threads>>>(n, input_data_A, input_data_B,
                                                          output_data);

  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int n = 1;

  dim3 blocks;
  dim3 threads;

  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;

  for (int i = 0; i < input->ndim; i++) {
      n *= input->shape[i];
  }

  if (n <= 1024) {
      blocks.x = 1;
      threads.x = n;
  } else {
      blocks.x = (n + 1023) / 1024;
      threads.x = 1024;
  }

  matrix_add_by_const_kernel<<<blocks, threads>>>(n, input_data, val,
                                                                 output_data);

  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int n = 1;

  dim3 blocks;
  dim3 threads;

  const float *input_data_A = (const float *)matA->data;
  const float *input_data_B = (const float *)matB->data;

  float *output_data = (float *)output->data;

  for (int i = 0; i < matA->ndim; i++) {
      n *= matA->shape[i];
  }

  if (n <= 1024) {
      blocks.x = 1;
      threads.x = n;
  } else {
      blocks.x = (n + 1023) / 1024;
      threads.x = 1024;
  }

  matrix_mul_kernel<<<blocks, threads>>>(n, input_data_A, input_data_B,
                                                          output_data);

  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int n = 1;

  dim3 blocks;
  dim3 threads;

  float *output_data = (float *)output->data;
  const float *input_data = (const float *)input->data;

  for (int i = 0; i < input->ndim; i++) {
      n *= input->shape[i];
  }

  if (n <= 1024) {
      blocks.x = 1;
      threads.x = n;
  } else {
      blocks.x = (n + 1023) / 1024;
      threads.x = 1024;
  }

  matrix_mul_by_const_kernel<<<blocks, threads>>>(n, input_data, val,
                                                         output_data);

  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here (Done) */
  // Hint: use cublas
  // cublas assume matrix is column major

  int rowA = matA->shape[0];
  int rowB = matB->shape[0];
  int colA = matA->shape[1];
  int colB = matB->shape[1];

  const int BLOCK_SIZE = 16;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((max(rowB, colB) + dimBlock.x - 1) / dimBlock.x,
          (max(rowA, colA) + dimBlock.y - 1) / dimBlock.y);

  const float *matA_data = (const float *)matA->data;
  const float *matB_data = (const float *)matB->data;
  float *matC_data = (float *)matC->data;
  
  MatMulKernel<<<dimGrid, dimBlock>>>(matA_data, matB_data, matC_data,
                                      rowA, rowB, colA, colB,
                                      transposeA, transposeB);

  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here (Done) */

  int n = 1;

  dim3 blocks;
  dim3 threads;

  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;

  for (int i = 0; i < input->ndim; i++) {
      n *= input->shape[i];
  }

  if (n <= 1024) {
      blocks.x = 1;
      threads.x = n;
  } else {
      blocks.x = (n + 1023) / 1024;
      threads.x = 1024;
  }

  relu_kernel<<<blocks, threads>>>(n, input_data, output_data);

  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here (Done) */
  int n = 1;

  dim3 blocks;
  dim3 threads;

  const float *input_data = (const float *)input->data;
  const float *input_grad_data = (const float *)in_grad->data;
  float *output_data = (float *)output->data;

  for (int i = 0; i < input->ndim; i++) {
      n *= input->shape[i];
  }

  if (n <= 1024) {
      blocks.x = 1;
      threads.x = n;
  } else {
      blocks.x = (n + 1023) / 1024;
      threads.x = 1024;
  }

  relu_gradient_kernel<<<blocks, threads>>>(n, input_data,
                                            input_grad_data, output_data);

  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here (Done)*/

  assert(input->ndim == 2);
  int nrow = input->shape[0];

  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];

  dim3 threads;

  const float *input_data = (const float *)input->data;
  float *output_data = (float *)output->data;

  if (nrow <= 1024) {
      threads.x = nrow;
  }
  else {
      threads.x = 1024;
      threads.y = (nrow + 1023) / 1024;
  }

  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_kernel<<<1, threads>>>(nrow, ncol, input_data, output_data);

  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
