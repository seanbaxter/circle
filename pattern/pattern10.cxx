#include <tuple>
#include <iostream>

int main() {
  // Use the parameter pack as a wildcard to consume the remainder of 
  // the binding elements.
  auto [x, y, ...] = std::make_tuple(1.1, 20, "Hello tuple", 'c');
  std::cout<< x<< " "<< y<< "\n";

  // Use single-element wildcards to space out the pack.
  auto [_, _, ...pack] = std::make_tuple(1.1, 20, "Hello tuple", 'c');
  std::cout<< "pack:\n";
  std::cout<< "  "<< pack<< "\n" ...;

  return 0;
}
