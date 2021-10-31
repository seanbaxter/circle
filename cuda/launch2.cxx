#include <cstdio>
#include <cuda_runtime_api.h>

template<auto x, typename type_t>
constexpr bool is_value_in_enum = (... || (@enum_values(type_t) == (type_t)x));

template<int x, int... y>
constexpr int upper_bound = x < y ...?? int... : sizeof...y;

enum class sm_selector : unsigned long long {
  sm_35 = 35, sm_37 = 37,
  sm_50 = 50, sm_52 = 52, sm_53 = 53,
  sm_60 = 60, sm_61 = 61, sm_62 = 62,
  sm_70 = 70, sm_72 = 72, sm_75 = 75,
  sm_80 = 80, sm_86 = 86,
};

// tuning params
using nt  [[attribute   ]] = int;
using vt  [[attribute(1)]] = int;
using occ [[attribute(0)]] = int;

// flags
using strided    [[attribute]] = void;
using persistent [[attribute]] = void;

template<typename tuning_t, typename func_t>
__global__ void launch_tuning_k(func_t func) {

  // Loop over all architectures specified at the compiler command line.
  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {

    // Enter the architecture being lowered to PTX.
    if target(arch == __nvvm_arch) {
      
      // Search for the best tuning for this architecture.
      constexpr int ub = upper_bound<(int)arch, tuning_t.enum_values...>;

      // There must be a viable tuning.
      static_assert(ub, "No viable tuning for " + arch.string);

      // Pluck out the best one.
      constexpr tuning_t tuning = @enum_value(tuning_t, ub - 1);

      // Report what we've chosen.
      @meta printf("Selecting tuning \"%s\" for arch %s\n", tuning.string, 
        arch.string);

      // Set the __launch_bounds__.
      __nvvm_maxntidx(@enum_attribute(tuning, nt));
      __nvvm_minctasm(@enum_attribute(tuning, occ));

      // Call the user function.
      func.template operator()<tuning>(threadIdx.x, blockIdx.x);
    }
  }
}

template<typename tuning_t, typename func_t>
void launch_tuning(const func_t& func, size_t count) {

  // Verify every tuning is supported in sm_selector.
  static_assert(
    is_value_in_enum<tuning_t.enum_values, sm_selector>,
    tuning_t.enum_names + " (" + ((int)tuning_t.enum_values).string + ") is invalid"
  )...;

  // Retrieve the kernel's arch version.
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, (const void*)&launch_tuning_k<tuning_t, func_t>);

  printf("Launching with PTX = sm_%d\n", attr.ptxVersion);

  // Get the best tuning for this arch.
  int index = attr.ptxVersion < (int)tuning_t.enum_values ...?
    int... - 1 : tuning_t.enum_count - 1;

  // Num values per block.
  int nt = 0, vt = 0;
  switch(index) {
    @meta for(int i : tuning_t.enum_count) {
      case i: // Use this tuning.
        nt = @enum_attribute(tuning_t, i, ::nt);
        vt = @enum_attribute(tuning_t, i, ::vt);
        break;
    } 
  }

  int nv = nt * vt;
  size_t num_blocks = (count + nv - 1) / nv;
  if(num_blocks)
    launch_tuning_k<tuning_t><<<num_blocks, nt>>>(func);
}

template<typename key_t>
void radix_sort(key_t* data, size_t count) {
  enum tuning_t {
    kepler  [[ .nt=128, .vt=5               ]] = 35,
    maxwell [[ .nt=256, .vt=7,  .persistent ]] = 52,
    pascal  [[ .nt=64,  .vt=11, .strided    ]] = 61,
    turing  [[ .nt=256, .vt=15, .occ=3      ]] = 75,
    ampere  [[ .nt=256, .vt=19, .strided    ]] = 86,
  };

  launch_tuning<tuning_t>([=]<tuning_t tuning>(int cta, int tid) {
    // This lambda is on the GPU.

    // Unpack the attributes.
    constexpr int nt = @attribute(tuning, ::nt);
    constexpr int vt = @attribute(tuning, ::vt);

    if(!cta && !tid) {
      // Let thread 0 print its tuning.
      printf("%s: sm_%d has %3dx%2d", tuning.string, 
        __builtin_current_device_sm(), nt, vt);
      if constexpr(@has_attribute(tuning, occ))
        printf(" occ=%d", @attribute(tuning, occ));
      if constexpr(@has_attribute(tuning, persistent))
        printf(" persistent");
      if constexpr(@has_attribute(tuning, strided))
        printf(" strided");
      printf("\n");

      printf("data = %p, count = %u\n", data, count);
    }

    // Allocate smem.
    __shared__ key_t shared[nt * vt];

    // Write each thread's ID down its smem lane with a compile-time loop.
    @meta for(int i : vt)
      shared[tid + i * nt] = tid;
    __syncthreads();

    // Or do the same thing with pack expansion.
    shared[tid + int...(vt) * nt] = tid ...;
    __syncthreads();

    // Pack expansion supports extended slices. Emit every other item in 
    // reverse order.
    shared[tid + int...(vt:0:-2) * nt] = tid ...;
    __syncthreads();

  }, count);
}

int main() {
  // Launch the kernel and synchronize to print.
  radix_sort<int>((int*)0xdeadbeef, 10101);
  cudaDeviceSynchronize();
}
