#include <cuda_runtime.h>
#include <algorithm>     // for std::upper_bound

template<nvvm_arch_t arch>
__global__ void kernel() {
  printf("Launched kernel<%s>().\n", @enum_name(arch));
}

int main() {
  // Get the PTX arch of the installed device.
  cudaDeviceProp prop { };
  cudaGetDeviceProperties(&prop, 0);
  int sm = 10 * prop.major + prop.minor;

  // Query the PTX of all targets being generated.
  const int targets[] { 
    (int)@enum_values(nvvm_arch_t)...
  };

  // Use upper_bound - 1 to find the largest PTX target not greater
  // than sm. This is what we'll be targeting.
  auto it = std::upper_bound(targets, targets + targets.length, sm);
  if(it == targets) {
    printf("No valid target for sm_%d.\n", sm);
    exit(1);
  }

  nvvm_arch_t target = (nvvm_arch_t)it[-1];
  printf("Selecting PTX target sm_%d.\n", target);

  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {
    if(arch == target) {
      // Conditionally launch a kernel template specialization over the
      // target architecture. Specify the PTX of the target to emit as the
      // 5th template parameter.
      kernel<arch><<<1, 1, 0, 0, (int)arch>>>();
    }
  }

  cudaDeviceSynchronize();
} 
