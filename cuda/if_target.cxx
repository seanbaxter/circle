#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel() {
  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {
    if target(arch == __nvvm_arch)
      printf("Compiling kernel for %s\n", arch.string);
  }
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}