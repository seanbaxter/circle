#include "../include/luacir.hxx"
#include "../include/format.hxx"
#include "params.hxx"

// Load and execute a Lua script.
@meta lua_engine_t kernel_lua;
@meta kernel_lua.file("kernel.lua");

template<int sm, typename type_t>
void fake_kernel(const type_t* input, type_t* output, size_t count) {
  // Look for a JSON item with the sm and typename keys.
  @meta kernel_key_t key { sm, @type_string(type_t) };
  @meta cirprint("Compiling kernel %:\n", key);

  // At compile-time, call the lua function kernel_params and pass our key.
  // If anything unexpected happens, we'll see a message.
  @meta params_t params = kernel_lua.call<params_t>("kernel_params", key);

  // Print the kernel parameters if they were found.
  @meta cirprint("  Params for kernel: %\n\n", params);
  
  // Off to the races--generate your kernel with these parameters.
}

int main(int argc, char** argv) {
  fake_kernel<52>((const float*)nullptr, (float*)nullptr, 0);
  fake_kernel<52>((const double*)nullptr, (double*)nullptr, 0);
  fake_kernel<70>((const float*)nullptr, (float*)nullptr, 0);
  fake_kernel<70>((const short*)nullptr, (short*)nullptr, 0);
  fake_kernel<35>((const int*)nullptr, (int*)nullptr, 0);
  return 0;
}
