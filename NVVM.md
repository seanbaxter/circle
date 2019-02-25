# Circle isn't CUDA

Integrating GPU kernel code with host code in the same source file presents us with a choice: 
1. Do make multiple front-end passes over the source and generate code for one target machine (i.e. either the host or any one of the GPU architecture versions) on each pass? This let's us assign different values to macros like `__CUDA_ARCH__` to influence the definitions depending on which target machine is being processed. This is flexible and simple, but results in long build times. This is the CUDA model. Both NVCC and Clang do this. 
1. Or do we somehow make a single front-end pass, and use multiple code generation passes to build target machine modules from a single AST? This is trickier, since we'll likely need to deal with `__CUDA_ARCH__` values explicitly as template arguments, since we can no longer keep redefining it as a preprocessor macro with every front-end pass. This is fast, but seemingly untenable.

Circle implements NVVM (CUDA) support with a single pass. It breaks the CUDA model. `__CUDA_ARCH__` will not be redefined with each architecural pass, because there is only a single pass. But it doesn't add any additional code partitioning burden to the programmer. It does, I think, something clever.

Values that guide compilation are necessarily constants. `__CUDA_ARCH__` and `__WARP_SIZE__` are constants. They're known during tokenization, actually, which is done before translation. Values like enums and constexpr objects are constants. They're known during translation. This is at definition (when the source is parsed) for non-dependent contexts, or possibly during instantiation for dependent contexts.

But there is another kind of compile-time constant: values that aren't known during tokenization, or during translation, but are known during code generation. If you only write programs for a single target machine, you probably have never seen any examples of these--why not just make that value a preprocessor constant? There's only one code generator pass, so it can only be one value. If you write CUDA code you're used to dealing with multiple target machines, but even then you've never seen an example of this, because CUDA makes one front-end pass for each target machine.

Circle's strategy for targeting multiple backends with a single front-end pass hinges on this: a codegen-time constant that contains device information and can be branched over to define target-dependent implementations. 

## Why this makes sense

CUDA code typically uses preprocessor directives to switch over `__CUDA_ARCH__` values:

```cpp
__device__ int reduction(int x, int tid) {
  int y = 0;

#if __CUDA_ARCH__ < 350
  // Do something appropriate for old devices

#elif __CUDA_ARCH__ < 600
  // Do somthing appropriate for Maxwell

#else
  // Do something appropriate for the new stuff

#endif
  return y;
}
```
Assuming we target three GPU device generations, `nvcc` will make four front-end passes over the translation unit: once for the host code and three times for the device code. During the device code passes, the `__device__`-tagged `reduction` function is parsed, and on each pass takes the one architecture branch appropriate for its `__CUDA_ARCH__`. This stops the code from using a too-new feature on a too-old device, or using a non-performant older feature on a device that supports more efficient mechanisms. If you're enforcing what features are actually available on each device architecture _during translation_, this is a good model.

Circle gets by with a single pass by exposing a codegen-time constant called `__nvvm_march` (or fully expanded, "NVIDIA Virtual Machine Machine Architecture"). We can no longer use preprocessor macros to branch on this value, because it's not known during tokenization. We can branch over it during translation, and that will add both the if and else statements to the AST. We'll then _evaluate the predicate_ during code-generation, when we know the values of `__nnvm_march` and the other codegen-time constants, and either emit or skip over that subtree in the AST for that machine target.

How do we do this safely? Introduce a new context on objects and expressions: the _codegen context_. This is similar to the constexpr and meta contexts in single-pass Circle which distinguish objects and expressions from ordinary AST constructs. When targeting NVVM, `__nvvm_march` is a codegen constant. If it's combined with a constexpr, meta or another codegen expression, the result object is also a codegen constant. If it's combined with a non-constant expression, that's an ill-formed statement.

When a codegen expression is used inside an if statement, it is guaranteed that the predicate will be evaluated during code generation and the branch not taken will not be emitted to the target module, and no contained declarations will be ODR-used. This is similar to an `if constexpr` construct, except the false branch is discarded during code generation rather than during template instantiation.

How do we make codegen-context object and if-statement declarations? With the `@codegen` extension, naturally. 

```cpp
__device__ int reduction(int x, int tid) {
  int y = 0;

  @codegen if(__nvvm_march < 35) {
    // Do something appropriate for old devices.

  } else @codegen if(__nvvm_march < 60) {
    // Do somthing appropriate for Maxwell.

  } else {
    // Do something appropriate for the new stuff.

  }
  return y;
}
```

If the predicate for a `@codegen if` isn't codegen itself, the program is ill-formed. If you use a codegen expression anywhere other than in the initializer of another codegen object or in a codegen if-statement, the program is ill-formed. This gives us a similar degree of security that the module for each target machine only includes the code that you intend for it to have, while at the same time allowing code for different target machines to co-exist during translation and enter the AST together.

## Go away, __device__ tag

