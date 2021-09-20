#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel() { 
  printf("Hello kernel, sm_%d.\n", __nvvm_arch);
}

int main() {
  // Launch only for sm_75.
  kernel<<<1, 1, 0, 0, 75>>>();
  cudaDeviceSynchronize();

  // Launch only for sm_35.
  kernel<<<1, 1, 0, 0, 35>>>();
  cudaDeviceSynchronize();
} 
