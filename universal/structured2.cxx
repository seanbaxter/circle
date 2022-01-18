#include <tuple>
#include <iostream>

int main() {
  // Declare a tuple-like object.
  std::tuple<double, int, const char*> tuple(3.14, 100, "Hello tuple");

  // Bind temporaries to its parts.
  auto [...parts] = tuple;

  // Print its components.
  std::cout<< sizeof. tuple<< " components:\n";
  std::cout<< int...<< ": "<< parts<< "\n" ...;
}