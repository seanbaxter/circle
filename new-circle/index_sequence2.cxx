#include <utility>
#include <iostream>

template<size_t N, typename F>
void func(F f) {
  // We can't do anything directly. Deduce the integer sequence into a
  // generate lambda. C++20 only.
  auto inner = []<size_t... Is>(F f, std::index_sequence<Is...>) {
    // Call f on each index Is. This is a Circleism.
    f(Is)...;
  };
  inner(f, std::make_index_sequence<N>());
}

int main() {
  func<5>([](size_t i) {
    std::cout<< "Got index "<< i<< "\n";
  });
}

