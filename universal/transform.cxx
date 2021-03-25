#include <tuple>
#include <iostream>

typedef float __attribute__((vector_size(16))) vec4;

template<typename a_t, typename b_t, typename func_t>
void binary_op(a_t& a, const b_t& b, func_t f) {
  f(a...[:], b...[:]) ...;
}

int main() {
  // Add a tuple with a vector.
  std::tuple<int, float, double, long> left(1, 2.f, 3.0, 4ll);
  vec4 right(1, 2, 3, 4);

  // Can pass a lambda and let an algorithm destructure the arguments.
  binary_op(left, right, [](auto& a, auto b) {
    a += b;
  });

  // Or just do it directly in line.
  left...[:] += right...[:] ...;

  std::cout<< int...<< ": "<< left...[:]<< "\n"...;
}

