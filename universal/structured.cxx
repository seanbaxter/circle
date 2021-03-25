#include <tuple>
#include <iostream>

int main() {
  // Declare a tuple-like object.
  std::tuple<double, int, const char*> tuple(3.14, 100, "Hello tuple");

  // Bind temporaries to its parts.
  auto [a, b, c] = tuple;

  // Print its components.
  std::cout<< "a: "<< a<< "\n";
  std::cout<< "b: "<< b<< "\n";
  std::cout<< "c: "<< c<< "\n";
}