#include <iostream>

template<typename... args_t>
void print_args(const args_t&... x) {
  // Note the trailing ... in the expression.
  std::cout<< x<< "\n" ...;
}

int main() {
  print_args(1, 3.14, "Hello");
  return 0;
}
