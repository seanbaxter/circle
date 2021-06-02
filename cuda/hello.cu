#include <cuda_runtime.h>
#include <cstdio>

__global__ void my_kernel() {
  int tid = threadIdx.x;
  printf("Hello CUDA %d.\n", tid);
}

int main() {
  my_kernel<<<1, 8>>>();
  cudaDeviceSynchronize();
}
