# Circle isn't CUDA

Circle adds GPU support but breaks with CUDA in two important ways:
1. The source is processed by the compiler front-end just once.
    This **tremendously** speeds up compilation, because CUDA parses the source once for each output target and an additional time for the host. Circle parses everything in one shot.
1. You can call any function from a kernel.  
    ... as long as it's defined in the translation unit you're calling it from. You'll only need to `__device__`-tag a function when it:
    * explicitly uses NVVM intrinsics, or
    * is intended to be called by device code in other translation units.  

Now you can directly use any types and functions from the standard headers in your device code. No more using forked-and-tagged versions of standard types that become stale with time. If your device code directly or indirectly calls a host function that does something prohibited, such as throw an exception, you'll still get a nice error message: but it is issued by Circle's code generator rather than its parser/semantic analyzer.

## Front-end passes

Integrating GPU kernel code with host code in the same source file presents us with a choice: 
1. Make _multiple front-end passes_ over the source.  
    Make a pass for each targeted device architecture and one more for the host. The compiler sets a different value to the `__CUDA_ARCH__` macro on each pass to help the source emit the best function definition for each architecture. This is flexible and simple, but results in long build times. This is the CUDA model. Both NVCC and Clang do this. 
1. Make a _single front-end pass_ over the source.  
     A single AST is built, and multiple IR modules are generated from this unified AST. This is trickier, since we'll likely need to deal with `__CUDA_ARCH__` values explicitly as template arguments, since we can no longer keep redefining it as a preprocessor macro with every front-end pass. This is fast, but _seemingly_ untenable. Or is it?

Circle implements NVVM (GPU) support with a single front-end pass. In other words, it breaks with the CUDA model: `__CUDA_ARCH__` will not be available to indicate the target device... Since there's only a single pass, each macro can only hold one value. But Circle achieves single-pass compilation without burdening the programmer. It does, I think, something clever.

Values that guide compilation are necessarily constants. `__CUDA_ARCH__` and `__WARP_SIZE__` are constants. As macros, they're known during tokenization. Values like enums and constexpr objects are constants. They're known during source translation.

There is another category of compile-time constant: values that aren't known during tokenization, or during translation, but are known during code generation. If you only write programs for a single target, you've probably never dealt with this distinction--I mean, why not just make that value a preprocessor constant? There's only one code generator pass, so it can only be one value.

If you write CUDA code you're used to dealing with multiple target machines, but even then you've never grappled with codegen constants, because CUDA makes a separate front-end pass for each target machine.

Circle's strategy for targeting multiple backends with a single front-end pass hinges on this: a codegen-time constant that contains device information and can be branched over to control target-dependent code. 

## Switching over architectures in CUDA

```cpp
template<int sm, typename type_t>
void radix_sort_sm(type_t* data, size_t count);

template<typename type_t>
void radix_sort(type_t* data, size_t count, int sm) {
  // Switch over the target machines and call to radix_sort_sm. 
  // Specialize on the sm version.
  if(sm < 52) {
    // Base-level hardware support. Require at least sm_35.
    radix_sort_sm<35>(data, count);

  } else if(sm < 60) {
    // Maxwell architecture support.
    radix_sort_sm<52>(data, count);
  
  } else if(sm < 70) {
    // Pascal support
    radix_sort_sm<60>(data, count);

  } else {
    // Latest-generation HW support.
    radix_sort_sm<70>(data, count);
  }
}
```
This snippet demonstrates a real irritation caused by CUDA's multi-pass model. We've written `__CUDA_ARCH__`-agnostic code, which ought to be a good thing (agnosticism usually brings portability to software). Unfortunately, the front-end gets run for each target machine, generating an M x N outer product of `radix_sort_sm` template instantiations. Each specialization is being generated each front-end pass. And these extra specializations are never discarded by the backend, even though they will never be called! `sm` is a runtime variable, so the _if-statement_ predicates won't be checked until runtime inside the host code. You end up with not just long build times but with bloated binaries.

I tried addressing this in the [moderngpu library](https://github.com/moderngpu/moderngpu/) with the [launch box](https://github.com/moderngpu/moderngpu/wiki/Launching-kernels#launch-box) mechanism. The launch box is an alchemical mixture of macro hacks and template metaprogramming. There are macros to map each `__CUDA_ARCH__` value to an `sm_ptx` typedef that gets built into every launch box and points the way to the parameter structure relevant for the GPU architecture that's targeted that pass.

[**kernel_sortedsearch.hxx**](https://github.com/moderngpu/moderngpu/blob/master/src/moderngpu/kernel_sortedsearch.hxx)
```cpp
template<bounds_t bounds, typename launch_arg_t = empty_t,
  typename needles_it, typename haystack_it, typename indices_it,
  typename comp_it>
void sorted_search(needles_it needles, int num_needles, haystack_it haystack,
  int num_haystack, indices_it indices, comp_it comp, context_t& context) {

  typedef typename conditional_typedef_t<launch_arg_t, 
    launch_box_t<
      arch_20_cta<128, 15>,
      arch_35_cta<128, 11>,
      arch_52_cta<128, 15>
    >
  >::type_t launch_t;

  ...

  // sm_ptx is defined to extract the parameters for the machine 
  // architecture corresponding to __CUDA_ARCH__.
  auto k = [=]MGPU_DEVICE(int tid, int cta) {
    typedef typename launch_t::sm_ptx params_t;
    enum { nt = params_t::nt, vt = params_t::vt, nv = nt * vt };

    // Do things with the parameter constants we've extract from the launch
    // box.
    __shared__ union {
      type_t keys[nv + 1];
      int indices[nv];
    } shared;
    ...
  };

  cta_transform<launch_t>(k, num_needles + num_haystack, context);
}
```
The launch box is a class template, and you use it by providing custom launch parameters wrapped in the `arch_{sm-version}` class template, or by providing thread- and value-count parameters in a more-convenient `arch_{sm-version}_cta` class template. These are usually defined inside host functions, where there is no `__CUDA_ARCH__` yet defined, so the launch box doesn't have a meaningful `sm_ptx` indicator yet. 

The kernel's body is defined inside a lambda function. This allows the kernel to capture the function parameters passed into the launcher. When the launcher's function template is instantianted, `cta_transform` receives the launch box for the kernel. Moderngpu's internals go to work, extracting the number of threads from the launch box, dividing the number of work items to get a grid size, and launching the kernel.

The crucial point is that there is no runtime switch over the SM version in this entire process. Macros are used to map `__CUDA_ARCH__` to an `sm_ptx` typedef, which guides the instantiation of the kernel's lambda function to the most appropriate parameter set for the target architecture.

Although it does what it set out to do, I can't say I loved this approach. It's too heavy weight and full of magic that's hard for me to follow, even though I wrote the thing. We can get rid of these incantitations by choosing a moving to a different compilation model.

## The codegen constant

CUDA code typically uses preprocessor directives to switch over `__CUDA_ARCH__` values:

```cpp
__device__ int reduction(int x, int tid) {
  int y = 0;

#if __CUDA_ARCH__ < 350
  // Do something appropriate for old devices

#elif __CUDA_ARCH__ < 600
  // Do something appropriate for Maxwell

#else
  // Do something appropriate for the new stuff

#endif
  return y;
}
```
Assuming we target three GPU device generations, `nvcc` will make four front-end passes over the translation unit: one for the host code and three for the device code. When the `__device__`-tagged `reduction` function is parsed, each pass takes a different branch in the `__CUDA_ARCH__`-predicated control flow. This prevents the code from using a too-new feature on a too-old device, or using a non-performant older feature on a device that supports more efficient mechanisms. If you're enforcing what features are actually available on each device architecture _during translation_, this is a good model.

Circle gets by with a single pass by exposing a codegen-time constant called `__nvvm_arch`. We can no longer use preprocessor directives to branch over this value, because the value is no longer known during tokenization. We can branch over it during translation (as if it was normal variable), and that will add both the if and else statements to the AST. We'll then _evaluate the predicate_ during code generation, when we know the values of `__nnvm_arch` and the other codegen-time constants, and either emit or skip over that subtree in the AST for that machine target.

How do we do this safely? Introduce a new context on objects and expressions: the _codegen context_. This is similar to the constexpr and meta contexts in Circle. When targeting NVVM, `__nvvm_arch` is a codegen constant. If it's combined with a constexpr, meta or another codegen expression, the result object is also a codegen constant. If it's combined with a non-constant expression, that's an ill-formed statement.

When a codegen expression is used inside an if statement, it is guaranteed that the predicate will be evaluated during code generation and only the branch taken will be emitted to the target module. This is similar to an `if constexpr` construct, except it's activated during code generation rather than template instantiation.

How do we make codegen-context object and _if-statement_ declarations? With the `@codegen` extension, naturally. 

```cpp
__device__ int reduction(int x, int tid) {
  int y = 0;

  @codegen if(__nvvm_arch < 35) {
    // Do something appropriate for old devices.

  } else @codegen if(__nvvm_arch < 60) {
    // Do something appropriate for Maxwell.

  } else {
    // Do something appropriate for the new stuff.

  }
  return y;
}
```
If the predicate for a `@codegen if` isn't codegen itself, the program is ill-formed. If you use a codegen expression anywhere other than in the initializer of another codegen object or in a codegen _if-statement_, the program is ill-formed. This provides a degree of security that the module for each target machine only includes code that you intend for it to have, while at the same time allowing code for different target machines to co-exist during translation and enter the AST together.

An added benefit is that we compile and test the semantic correctness of all versioned code, even if we're not targeting that architecture. Often during development, to keep build times acceptible, you target only the architecture of the card you have in your system. Then later when you make a full build you're greeted with a lot of errors enabled by the new values of `__CUDA_ARCH__`. The codegen constant system catches those compiler errors early.

It's during code-generation and not during translation where we check for the availability of a feature for the device architecture being targeted for that LLVM module. For instance, a feature for Pascal architecture must be guarded with codegen control-flow if it's used in a function that's also ODR-used in the backend pass for the IR module for the Maxwell architecture. We _don't_ need to guard these features generally, but only when the containing function is either non-inline or ODR-used from device code used from an earlier architecture. (See [Calling untagged functions from device code](#calling-untagged-functions-from-device-code) for a discussion on how ODR-usage drives code generation.)

## Switching over architectures in Circle

```cpp
// Define the supported architectures in this build. The code generator will 
// build a module for each of these. The enum names are mostly for fun--what
// we really care about are the values.
enum class gpu_arch_t {
  kepler = 35,
  maxwell = 52,
  pascal = 60,
  volta = 70,
  turing = 75,
};

template<gpu_arch_t arch, typename type_t>
void radix_sort_sm(type_t* data, size_t count) {
  // There's just one definition of the radix_sort_sm function template. It's 
  // specialized over the sm number, but since there's just one front-end pass,
  // it's effectively seen by all the sm versions.

  // Use the JSON-loading trickery in the Querying JSON general example.

  // Look for a JSON item with the sm and typename keys.
  @meta kernel_key_t key { (int)arch, @type_name(type_t) };

  // At compile-time, find the JSON item for key and read all the members
  // of params_t. If anything unexpected happens, we'll see a message.
  @meta params_t params = find_json_value<params_t>(kernel_json, key);

  // Generate the function definition using the constants in params.
}

template<typename type_t>
void radix_sort(type_t* data, size_t count) {
  // Switch over the target machines and call to radix_sort_sm. 
  // Specialize on the sm version.
  @meta for(int i = 0; i < @enum_count(gpu_arch_t); ++i)
    @codegen if((int)@enum_value(gpu_arch_t, i) == __nvvm_arch)
      radix_sort_sm<@enum_value(gpu_arch_t, i)>(data, count);
}
```
Define an enum with one enumerator per target machine architecture. Call it `gpu_arch_t`. In the futureu, Circle might define this automatically for nvvm builds.

Write a function template that defines the kernel and parameterize it over a `gpu_arch_t` enum. Following the general examples [Querying JSON](https://github.com/seanbaxter/circle#querying-json) and [Querying Lua](https://github.com/seanbaxter/circle#querying-lua), we employ meta statements to retrieve tuning constants from the target machine version and the function parameter types. The flashiest solution is to put the parameters in a separate file, but you can also define them in any ordinary data structure using a meta object in namespace scope. For example, 
```cpp
@meta std::map<kernel_key_t, params_t> radix_sort_params_map {
  // Construct this map with key/value pairs for the radix sort.
};
```
is a convenient way to make parameters indexable by kernel key. When source translation is done, the `radix_sort_params_map` is deleted with all other meta objects of static storage duration. It will not be included in your binary.

[Enum introspection](https://github.com/seanbaxter/circle#introspection-on-enums) is the all-star feature in Circle, and it's used again by the `radix_sort` launcher. The metafor steps over all `gpu_arch_t` values. At each step, a codegen context _if-statement_ matches the sm version of the enumerator to the codegen constant `__nvvm_arch` and emits a call to `radix_sort_sm` specialized over that SM version. We don't know what value `__nvvm_arch` has at translation, but that's okay--Circle makes only a single front-end pass over the translation unit, so no unwanted code will be added to the AST. 

When the code generator runs for each GPU architecture, only the child statement of the `@codegen if` statement that matches the targeted machine architecture gets emitted as NVVM code.

If it's too burdensome to provide a parameter set for every GPU architecture being targeted, this code can easily be adjusted to match the `__nvvm_arch` to a range of targets covered by a single parameter set. That would generate M (the number of different tunings) function template instantiations and N (the number of machine targets) backend versions. Each of the N target machines uses the most appropriate of the M tunings.

Is this actually simpler than the launch box solution? Yes, very much so. The Circle snippet shows all the code required to map a device architecture to a template instantiation. The moderngpu snippet relies on very complex tooling hidden away in headers. The launch box tooling may work for you, but when it doesn't, the transparency of Circle's approach makes it easy to write your own kernel-launch logic.

## Calling untagged functions from device code

CUDA only allows kernels to call functions marked with `__device__`, and `__device__` functions can only call other `__device__` functions. This is very bad. (Although recent versions of nvcc do have an experimental --expt-relaxed-constexpr flag to allow calling constexpr functions from device functions.) Large amounts of the STL had to be ported into thrust, duplicating hundreds of data types and functions, just to add `__device__` tokens everywhere.

Circle retains `__device__` tags, but only needs them on function declarations that intend to be shared across translation units. Locally-defined functions, which includes all inline functions (almost all functions these days), do not need the tags. Here's what happens:

Start the code generator at a kernel function. This still gets tagged with the `__global__` token. The backend traverses the kernel's AST and emits NVVM IR. When it encounters a function call, there are a few things that can happen:
1. If the function is externally defined and marked `__device__`, the backend emits a call instruction. It's then up to the linker to find a definition.
1. If the function is externally defined and not marked `__device__`, the program is ill-formed.
1. If the function is locally defined, we mark it ODR-used and add it to a queue of function definitions that require NVVM code generation. This happens regardless of if the function is `__device__`-tagged or not.

NVVM IR for non-inline functions marked `__device__` are always added to the module, even if the functions aren't ODR-used from inside the translation unit. The expectation is that they'll be called from other translation units.

When the code generator builds NVVM IR for untagged functions, usage of prohibited features causes a compile-time error. For example, _throw-expressions_ are supported in host code but not in device code, so using one generates a compiler error. Consequently, _try-blocks_ in host functions are eliminated and replaced by the primary _compound-statement_; since there are no exceptions, all the catch handlers are swept away and we don't ODR-use any of their dependencies.

Untagged inline functions that are ODR-used (directly or indirectly) from a kernel exhibit the `linkonce_odr` attribute on their IR output. This is consistent with any other inline function. The linker chooses one instance of the function definition from modules _for the same device architecture_ across all translation units.

