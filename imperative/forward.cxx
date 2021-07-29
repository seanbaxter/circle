#include <type_traits>
#include <utility>
#include <iostream>

template<typename... Ts>
void func(Ts&&... x) {
  std::cout<< int...<< ": "<< x<< "\n" ...;
}

template<typename... Ts>
void forward_ints(Ts&&... x) {
  // Use .remove_reference to account for lvalue and rvalue references.
  func(if Ts.remove_reference.is_integral => std::forward<Ts>(x) ...);
}

int main() {
  int a = 1;
  forward_ints(a, 'b', 3.3, "Four", nullptr, 6ull);
}