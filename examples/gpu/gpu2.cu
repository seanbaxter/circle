#include <cuda.h>
#include <cuda_runtime.h>
#include <optional>
#include <cstdio>

// Print any enum.
template<typename type_t>
const char* name_from_enum(type_t e) {
  static_assert(std::is_enum<type_t>::value);

  switch(e) {
    @meta for enum(type_t e2 : type_t)
      case e2:
        return @enum_name(e2);

    default:
      return nullptr;
  }
}

// Find the most compatible compiled architecture for this device.
// This maps a runtime value (the compute capability from the driver) to 
// a copmile-time enumerator from nvvm_arch_t, which is 
std::optional<nvvm_arch_t> find_best_arch(int device = 0) {
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  int sm = 10 * major + minor;

  // nvvm_arch_t is an implicitly-defined scoped enum holding all NVPTX targets
  // for the translation unit.
  std::optional<nvvm_arch_t> best;
  @meta for(int i = 0; i < @enum_count(nvvm_arch_t); ++i) {
    if(sm >= (int)@enum_value(nvvm_arch_t, i))
      best = @enum_value(nvvm_arch_t, i);
  }

  return best;
}

// Generic kernel that instantiates its function object with the 
// nvvm_arch_t (__nvvm_arch) currently being targeted by the LLVM backend.
template<int nt, typename kernel_t>
__global__ void sm_launch(kernel_t kernel) {
  // Specialize the kernel's function template over each possible 
  // nvvm arch. The @codegen if is evaluated during code generation and only
  // emits code for the architecture being targeted by that module.
  @meta for enum(nvvm_arch_t sm : nvvm_arch_t) {
    @codegen if(sm == __nvvm_arch)
      kernel.template go<sm, nt>();
  }
}

// You define this bit.
struct my_kernel_t {
  template<nvvm_arch_t sm, int nt>
  __device__ void go() {
    // Look up kernel parameters by sm, parameter type, etc. Drive the kernel
    // definition using a combination of meta and codegen control flow.
  }
};


int main() {
  // Circle enums are iterable collections. Print all the architectures 
  // targeted in this build. That's one for each -sm_XX command-line argument.
  @meta printf("Device architectures targeted in this build:\n");
  @meta for enum(nvvm_arch_t sm : nvvm_arch_t)
    @meta printf("  %s\n", @enum_name(sm));

  // Print the best targeted architecture for this device.
  nvvm_arch_t arch;
  if(auto best = find_best_arch()) {
    arch = *best;
    printf("Selected device architecture %s\n", name_from_enum(arch));

  } else {
    printf("Invalid device version for these build parameters\n");
    exit(1);
  }

  // Prepare a kernel. The go function template will be instantiated with 
  // the value of __nvvm_arch for each backend and the block size.
  my_kernel_t my_kernel { };

  const int nt = 128;
  sm_launch<nt><<<1, nt>>>(my_kernel);

  return 0;
}