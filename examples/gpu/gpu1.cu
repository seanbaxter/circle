#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <vector>

// Perform an inclusive prefix scan with one input per thread.
template<int nt, typename type_t, typename op_t = std::plus<type_t> >
__device__ type_t cta_scan(int tid, type_t x, op_t op = op_t()) {

  // Provision 2 * nt shared memory slots for double-buffering. This cuts in
  // half the number of __syncthreads required.
  __shared__ type_t shared[2 * nt];

  int first = 0;
  shared[first + tid] = x;
  __syncthreads();

  @meta for(int offset = 1; offset < nt; offset *= 2) {
    // Increment using the element on the left.
    if(tid >= offset)
      x = op(shared[first + tid - offset], x);

    // Write back to tid's slot in the double buffer.
    first = nt - first;
    shared[first + tid] = x;
    __syncthreads();
  }

  // Return the accumulated value.
  return x;
}

// Kernels are still marked __global__.
template<int nt>
__global__ void my_kernel(int* p) {
  int tid = __nvvm_tid_x();
  int x = cta_scan<nt>(tid, 1, std::plus<int>());
  p[tid] = x;
}

int main() {
  const int nt = 64;
  int* data;
  cudaMalloc(&data, nt * sizeof(int));

  my_kernel<nt><<<1, nt>>>(data);

  std::vector<int> results(nt);
  cudaMemcpy(results.data(), data, nt * sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < nt; ++i)
    printf("%d -> %d\n", i, results[i]);

  cudaFree(data);

  return 0;
}