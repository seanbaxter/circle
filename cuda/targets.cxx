#include <iostream>

int main() {
  // Print all PTX targets.
  std::cout<< @enum_names(nvvm_arch_t)<< " = "
    << (int)@enum_values(nvvm_arch_t)<< "\n" ...;
}

