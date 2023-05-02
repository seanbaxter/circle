#feature on recurrence
#include <algorithm>
#include <iostream>

auto reduce(const auto& init, const auto&... args) {
  // Equivalent to a left-binary fold:
  // As if
  //   return (init + ... + args);
  // Expands to init + args#0 + args#1 + args#2 etc.
  return (recurrence init + args ...);
}

// f is any callable.
auto fold(auto f, const auto& init, const auto&... args) {
  // Equivalent to a left-binary fold, but calls a function!
  // As if 
  //   return (init f ... f args);
  // were supported.

  // Expands to:
  //   f(f(f(init, args#0), args#1), args#2) etc
  return (f(recurrence init, args) ...);
}

int main() {
  int data[] { 2, 3, 4, 10, 20, 30 };

  // Send a pack of arguments to reduce.
  auto sum = reduce(1, data...);
  std::cout<< "sum: "<< sum<< "\n";

  // Reduce with a lambda function.
  auto product = fold([](auto x, auto y) { return x * y; }, 1, data...);
  std::cout<< "product: "<< product<< "\n";
}
