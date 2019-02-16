#include <fstream>
#include "../include/jsoncir.hxx"
#include "params.hxx"

// Load a JSON file.
@meta json kernel_json;
@meta std::ifstream json_file("kernel.json");
@meta json_file>> kernel_json;

template<int sm, typename type_t>
void fake_kernel(const type_t* input, type_t* output, size_t count) {
  // Look for a JSON item with the sm and typename keys.
  @meta kernel_key_t key { sm, @type_name(type_t) };
  @meta cirprint("Compiling kernel %:\n", key);

  // At compile-time, find the JSON item for key and read all the members
  // of params_t. If anything unexpected happens, we'll see a message.
  @meta params_t params = find_json_value<params_t>(kernel_json, key);

  // Print the kernel parameters if they were found.
  @meta cirprint("  Params for kernel: %\n\n", params);
  
  // Off to the races--generate your kernel with these parameters.
}

int main() {
  fake_kernel<52>((const float*)nullptr, (float*)nullptr, 0);
  fake_kernel<52>((const double*)nullptr, (double*)nullptr, 0);
  fake_kernel<70>((const float*)nullptr, (float*)nullptr, 0);
  fake_kernel<70>((const short*)nullptr, (short*)nullptr, 0);
  fake_kernel<35>((const int*)nullptr, (int*)nullptr, 0);
  return 0;
}