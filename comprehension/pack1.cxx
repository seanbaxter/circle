#include <vector>
#include <cstdio>

inline int sq(int x) {
  return x * x;
}

int main() {
  std::vector x = [ @range(10)... ];      // shorthand for @range(0:10:1).
  std::vector y = [ @range(10::-1)... ];  // shorthand for @range(10:0:-1).

  // Compute element-wise sq(x) + 5 * y, 
  std::vector z = [ sq(x[:]) + 5 * y[:] ... ];

  // Print each element.
  printf("sq(%d) + 5 * %d -> %2d\n", x[:], y[:], z[:]) ...;
}