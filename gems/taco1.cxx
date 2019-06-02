#include "util.hxx"
#include <taco.h>

@meta int taco_kernel_count = 0;

template<typename... args_t>
@meta void call_taco(@meta const char* pattern, args_t&&... args) {

  // Run the taco system call.
  @meta std::string s = format("taco \"%s\" -print-nocolor", pattern);
  @meta std::string code = capture_call(s.c_str());

  // Print the code we're injecting.
  @meta printf("%s\n", code.c_str());

  // taco always returns kernels named "kernel". To avoid name collisions, 
  // create a new namespace. Use the global counter taco_kernel_count.
  @meta std::string ns_name = format("taco_kernel_%d", ++taco_kernel_count);

  // Inject the code into a custom namespace.
  @statements namespace(@(ns_name))(code, "taco kernel");

  // Call the function. Pass each argument tensor by address.
  @(ns_name)::compute(&args...);
}

int main() {
  taco_tensor_t v { }, M { }, x { }, N { };

  // inject taco_kernel_1::compute
  call_taco("v(i) = M(i, j) * x(j)", v, M, x);

  // inject taco_kernel_2::compute
  call_taco("v(i) = M(i, j) * N(j, k) * x(k)", v, M, N, x);

  return 0;
}