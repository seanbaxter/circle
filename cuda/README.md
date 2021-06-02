# CUDA on Circle

Circle's [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) support is a work in progress. The compiler is being modified to track changes to the thrust, CUB and CUDA Toolkit libraries as they appear.

Circle is a single-pass heteregeneous compiler. It already targets [single-source shaders](https://github.com/seanbaxter/shaders/blob/master/README.md) using the SPIR-V and DXIL intermediate representations with a single translation pass. CUDA support adds a new PTX and SASS target.

While the `__host__` and `__device__` tags are still supported in part, they you aren't required to tag functions to call them from kernels. This makes Standard Library code available for execution on the GPU.

## Usage

To compile for CUDA, use these command-line options:
* `--cuda-path=<path-to-toolkit>` - You must point the compiler at a recent installation of the CUDA Toolkit. A `.cu` file extension is not required. The presence of `--cuda-path` enables CUDA compilation.
* `-sm_XX` - One or more PTX target architectures are required. No default architecture is assumed. Supported values are 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, and 75. 80 and 86 will be supported when Circle bumps LLVM from 8 to 11.
* `-gpu` - An optional argument to indicate that a .ll or .bc output (LLVM IR and bitcode) should relate to the GPU code and not the host code. For binary outputs, where fatbinary is implicitly included into the executable, this flag may not be used.
* `-G` - Enable debug information on device targets.

[**hello.cu**](hello.cu)
```cpp
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
```
```
$ circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 -sm_75 hello.cu -lcudart -o hello
$ ./hello
Hello CUDA 0.
Hello CUDA 1.
Hello CUDA 2.
Hello CUDA 3.
Hello CUDA 4.
Hello CUDA 5.
Hello CUDA 6.
Hello CUDA 7.
```

Be sure to specify `-lcudart` to link with the CUDA Runtime library `libcudart.so`.

As with the `nvcc` compiler, Circle locates the compiled kernel modules in the `.nv_fatbin` data section. This allows CUDA tooling like `cuobjdump` to locate and print the kernels:

```
$ cuobjdump -lptx -lelf hello
ELF file    1: hello.1.sm_52.cubin
PTX file    1: hello.1.sm_52.ptx
ELF file    2: hello.2.sm_75.cubin
PTX file    2: hello.2.sm_75.ptx
```

## Reflection and _if-target_

Circle implicitly defines an enumeration `nvvm_arch_t` populated with the PTX architectures specified on the command line.

[**targets.cxx**](targets.cxx)
```cpp
#include <iostream>

int main() {
  // Print all PTX targets.
  std::cout<< @enum_names(nvvm_arch_t)<< " = "
    << (int)@enum_values(nvvm_arch_t)<< "\n" ...;
}
```
```
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 targets.cxx && ./targets
sm_35 = 35
sm_52 = 52
sm_61 = 61
sm_75 = 75
```

Each enumerator is named `sm_XX`, where XX is the PTX version, which is also its integral value. Since Circle supports [static reflection](https://github.com/seanbaxter/circle/blob/master/reflection/README.md), you can programmatically interact with the definition of this enumeration.

[**if_target.cxx**](if_target.cxx)
```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel() {
  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {
    if target(arch == __nvvm_arch)
      printf("Compiling kernel for %s\n", @enum_name(arch));
  }
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```
```
$ circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 if_target.cxx -lcudart && ./if_target 
Compiling kernel for sm_75
```

Circle implicitly declares a _codegen_ object of type `nvvm_arch_t` called `__nvvm_arch`. The value of this object is available during code generation. Circle makes a single frontend translation pass and builds a single AST that describes the entire program. It makes a code-generation pass for each target, be it generic/host, SPIR-V, DXIL, or any of the PTX targets. The predicate expressions inside _if-target_ statements are evaluated during code generation. The true branch is emitted to the binary, and all false branches are discarded. This is similar to _if-constexpr_, but deferred until code generation.

Use reflection on `nvvm_arch_t` to visit each PTX target and conditionally emit definitions specific to each target. The compilation of `if_target.cxx` lowers `kernel` to four PTX targets. The definition of `kernel` includes a single architecture-specific `printf` statement. The branch is executed during code generation, not during runtime.

## PTX launch chevrons

CUDA compilers support a chevron for launching kernels. This has four arguments, and the latter two have zero defaults:
* grid size.
* block size.
* dynamically-provisioned device shared memory in bytes.
* `cudaStream_t` object to execute on a stream.

As an alternative to the _if-target_ method of dispatch, Circle has extended the launch chevrons by adding an optional fifth argument:
* PTX target to ODR-use. This must be a constant integral expression with the value of the one of PTX targets specified at the command line.

During code generation, launching a kernel or taking its address ODR-uses that kernel from each of the PTX targets. This is desired, except when specializing a kernel function template with an architecture-specific tuning. If we specialize `kernel<sm_52>`, we obviously intend to only call that from the PTX 52 architecture, and not from any others. However, the PTX and SASS for that function specialization would be generated for all PTX targets, resulting in long build times and an executable that ships with a lot of unreachable code.

[**launch.cxx**](launch.cxx)
```cpp
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
```
```
$ circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 launch.cxx -lcudart  && ./launch
Selecting PTX target sm_75.
Launched kernel<sm_75>().
```

This sample uses a different mechanism, the PTX launch argument, to achieve the same effect as _if-target_-driven function definitions. The first task, which needs only be executed once at the start of the program, is to map the _binaryVersion_ of the device (i.e. the architecture version that the GPU itself implements) to the PTX version candidate it will draw its kernel definitions from.

```cpp
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
```

It is easiest to use the pack-yielding type introspection keyword `@enum_values` to expand the candidate architectures into an integer array, and then to use `std::upper_bound` to search for the `binaryVersion` within the sorted array of `ptxVersion`s. If result points to the start of the array, execution fails, because the earliest PTX version comes after the device's architecture. Otherwise, the `ptxVersion` to use is one before the upper bound.

```cpp
  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {
    if(arch == target) {
      // Conditionally launch a kernel template specialization over the
      // target architecture. Specify the PTX of the target to emit as the
      // 5th template parameter.
      kernel<arch><<<1, 1, 0, 0, (int)arch>>>();
    }
  }
```

Static reflection visits each PTX target candidate. An ordinary _if-statement_ tests if this is the `ptxVersion` selected for the current device. If it is, the kernel is launched, with the designated PTX version provided as the 5th chevron argument. This suppresses generation of device code for this kernel for all architectures except the one specified. This conditional code generation allows us to specialize the kernel function template on a PTX-dependent tuning directly, and even mark up the kernel's declaration with `__launch_bounds__`. This is not possible with a 4-argument launch chevron without emitting a quadratic number of unreachable kernel definitions.

```cpp
#include <cuda_runtime.h>

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
```
```
$ circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 bad_launch.cxx -lcudart -g && ./bad_launch
Hello kernel, sm_75.
bad_launch: /home/sean/projects/circle/cuda/bad_launch.cxx:13: int main(): Assertion `contract violation launching CUDA kernel: PTX version 35 does not match device architecture.' failed.
Aborted (core dumped)
```

It's up to the caller to only perform a 5-argument chevron launch where the PTX argument matches what the CUDA runtime would select for the current device. Blindly attempting to launch the kernel may or may not result in the kernel's execution, depending on if the kernel was ODR-used by another architecture at some point during code generation. To protect against these mistakes, compile your program with the `-g` command line option which enables asserts. When `NDEBUG` is not defined (meaning we are compiling with asserts), the Circle backend calls `cudaFuncGetAttributes` and compares the `cudaFuncAttributes::ptxVersion` member against the fifth chevron argument. If `cudaFuncGetAttributes` fails, or if the two PTX versions don't match, an assert is raised to indicate a contract violation.

## User attributes for tuning

[**tuning1.cxx**](tuning1.cxx)
```cpp
#include <cstdio>

template<auto x, typename type_t>
constexpr bool is_value_in_enum = (... || (@enum_values(type_t) == (type_t)x));

// Set of device architectures. Will be part of CUDA Toolkit.
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

// Tunings for a specific operation.
enum class tuning_t {
  kepler  [[ .nt=128, .vt=5               ]] = 35,
  maxwell [[ .nt=256, .vt=7,  .persistent ]] = 52,
  pascal  [[ .nt=64,  .vt=11, .strided    ]] = 61,
  turing  [[ .nt=256, .vt=15, .occ=3      ]] = 75,
  ampere  [[ .nt=256, .vt=19, .strided    ]] = 86,
};

// Test that each tuning corresponds to an actual device architecture.
static_assert(
  is_value_in_enum<@enum_values(tuning_t), sm_selector>,
  @string(@enum_names(tuning_t), " (", (int)@enum_values(tuning_t), ") is invalid")
)...;

int main() { 
  // Print the tunings using a loop.
  printf("With a loop:\n");
  @meta for enum(tuning_t tuning : tuning_t) {
    printf("%-10s: %3dx%2d\n", @enum_name(tuning),
      @enum_attribute(tuning, nt), @enum_attribute(tuning, vt));
  }

  // Print the tunings using pack expansion.
  printf("\nWith a pack expansion:\n");
  printf("%-10s: %3dx%2d\n", @enum_names(tuning_t), 
    @enum_attributes(tuning_t, nt), @enum_attributes(tuning_t, vt)) ...;
}
```
```
$ circle tuning1.cxx && ./tuning1
With a loop:
kepler    : 128x 5
maxwell   : 256x 7
pascal    :  64x11
turing    : 256x15
ampere    : 256x19

With a pack expansion:
kepler    : 128x 5
maxwell   : 256x 7
pascal    :  64x11
turing    : 256x15
ampere    : 256x19
```

Circle supports [user-defined attributes](https://github.com/seanbaxter/circle/blob/master/reflection/README.md#user-attributes), which can be queried with the compiler's static reflection keywords. User attributes may be the most convenient way to specify the tuning parameters that govern kernel generation. The `tuning_t` enumeration pins user attributes on each enum. The `@enum_attribute` extension retrieves the compile-time attribute off each enumerator. The `@enum_has_attribute` extension tests the presence of an attribute, which is useful for flag attributes, like `strided` and `persistent`.

## Powering _if-target_ kernels with user attributes

[**launch2.cxx**](launch2.cxx)
```cpp
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
      printf("%s: sm_%d has %3dx%2d", @enum_name(tuning), 
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
```

This sample demonstrates _if-target_ based kernel generation driven by tuning attributes. `launch_tuning` is a function template that accepts a tuning _enumeration_ (meaning, a collection of enumerators, each representing one PTX version) as a template argument, maps each PTX target (those specified at the command line) with a tuning, launches a generic CUDA kernel with the best tuning, and calls back into the caller-provided lambda, which becomes the body of the kernel definition.

Inside the body of the lambda, the user can access tuning attributes with the `@attribute` and `@has_attribute` Circle extensions on the explicit template argument `tuning`.  

```cpp
template<auto x, typename type_t>
constexpr bool is_value_in_enum = (... || (@enum_values(type_t) == (type_t)x));

template<typename tuning_t, typename func_t>
void launch_tuning(const func_t& func, size_t count) {

  // Verify every tuning is supported in sm_selector.
  static_assert(
    is_value_in_enum<@enum_values(tuning_t), sm_selector>,
    @string(@enum_names(tuning_t), " (", (int)@enum_values(tuning_t), ") is invalid")
  )...;

  // Retrieve the kernel's arch version.
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, (const void*)&launch_tuning_k<tuning_t, func_t>);

  printf("Launching with PTX = sm_%d\n", attr.ptxVersion);

  // Get the best tuning for this arch.
  int index = attr.ptxVersion < (int)@enum_values(tuning_t) ...?
    int... - 1 : @enum_count(tuning_t) - 1;

  // Num values per block.
  int nt = 0, vt = 0;
  switch(index) {
    @meta for(int i : @enum_count(tuning_t)) {
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
```

`launch_tuning` has two parts: the host function, listed above, and the kernel entry point, listed below. On the host side, we perform a sanity check, confirming that each tuning in the collection corresponds to an actual device architecture listed in the Toolkit-provided `sm_selector` enum. If the user provides a tuning for, say, sm_63, that is a compile-time error, because sm_63 does not exist.

Next, query `cudaFuncGetAttributes` for the PTX version of the templated entry point specialized over the collection of tuning parameters, `tuning_t`. Taking the address of a kernel ODR-uses it from all PTX targets, so this query operation should always succeed for the current device. The challenge now is to find the tuning that best corresponds to the PTX target of the current device.

We use the same _upper-bound - 1_ technique to find the best tuning for a PTX target. The [Circle multi-conditional operator ...?](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#multi-conditional---) lets us express this as a one-liner. The upper bound is the index of the first element in a sorted collection that is greater than a key. It's the first index in the pack expansion for which `attr.ptxVersion < (int)@enum_values(tuning_t)` is true. `int...` is an expression that yields the current pack index.

Now that we know the index of the tuning, use reflection to visit each tuning enumerator. We enter a switch statement with the tuning index as the predicate expression. A _meta-for_ generates the switch cases for each tuning. When the runtime tuning index `index` matches the compile-time tuning index `i`, the `nt` and `vt` user-defined attributes are extracted from the `i`th tuning enumerator. These signify the block size ("num threads") and grain size ("values per thread") respectively. Their product is the number of data items retired by each CUDA CTA. The host part of the CUDA dispatch finishes by launching enough blocks to cover the input count. Note that the `launch_tuning_k` kernel entry point is specialized over the enumeration `tuning_t`, which holds all tunings, and not a particular tuning. We don't want anything PTX-version specific to be represented in the kernel declaration, because that would generate unreachable kernel definitions that slow your build and bloat the binary.

```cpp
template<int x, int... y>
constexpr int upper_bound = x < y ...?? int... : sizeof...y;

template<typename tuning_t, typename func_t>
__global__ void launch_tuning_k(func_t func) {

  // Loop over all architectures specified at the compiler command line.
  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {

    // Enter the architecture being lowered to PTX.
    if target(arch == __nvvm_arch) {
      
      // Search for the best tuning for this architecture.
      constexpr int ub = upper_bound<arch, @enum_values(tuning_t)...>;

      // There must be a viable tuning.
      static_assert(ub, @string("No viable tuning for ", @enum_name(arch)));

      // Pluck out the best one.
      constexpr tuning_t tuning = @enum_value(tuning_t, ub - 1);

      // Report what we've chosen.
      @meta printf("Selecting tuning \"%s\" for arch %s\n", @enum_name(tuning),
        @enum_name(arch));

      // Set the __launch_bounds__.
      __nvvm_maxntidx(@enum_attribute(tuning, nt));
      __nvvm_minctasm(@enum_attribute(tuning, occ));

      // Call the user function.
      func.template operator()<tuning>(threadIdx.x, blockIdx.x);
    }
  }
}
```

The kernel entry point implements an _if-target_ switch to generate target-specific kernel definitions from a single template specialization. We start by looping over all PTX targets specified on the command line. This is different from the use of reflection on the host side, which loops over all tunings in the tuning collection. The two collections overlap, but are not coincident.

The _if-target_ predicate `arch == __nvvm_arch` is true when the Circle backend lowers the AST for the kernel to NVVM IR for the `arch` PTX version. We perform the compile-time upper-bound search to get the user-provided tuning out of the `tuning_t` collection that best fits the current PTX target. Unlike in the host, where the PTX version was a runtime variable retrieved with `cudaFuncGetAttributes`, inside the kernel it's a compile-time constant, `arch`, retrieved using reflection on the target enumeration `nvvm_arch_t`. 

```cpp
template<int x, int... y>
constexpr int upper_bound = x < y ...?? int... : sizeof...y;
```

The `upper_bound` variable template is a one-liner that exploits the [constexpr multi-conditional operator ...??](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-multi-conditional---), which is unique to Circle. It computes the upper bound given a compile-time key (`arch`) and a sorted list of compile-time values (`@enum_values(tuning_t)`).

After selecting the best tuning given the backend's `__nvvm_arch` value, we call the `__nvvm_maxntidx` and `__nvvm_minctasm` compiler intrinsics to set the kernel's launch bounds. The `__launch_bounds__` CUDA is not compatible with _if-target_ based kernel generation, because it marks the kernel's declaration, which would require the user to specialize the kernel template over a particular tuning and result in a quadratic generation of PTX code, with many unreachable instances in the fatbin. By exposing the launch bounds functionality with intrinsics invoked inside the definition, we can guard the launch bounds inside the _if-target_ switch.

Finally, the user's function object is invoked (on the device side) and the callback function is specialized over the backend's PTX target tuning.
