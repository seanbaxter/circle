#include <utility>
#include <iostream>

template<typename F, size_t... Is>
void func_inner(F f, std::index_sequence<Is...>) {
  // Call f on each index Is. This is a Circleism.
  f(Is) ...;
}

template<size_t N, typename F>
void func(F f) {
  // we can't do anything with N here. We have to deduce the integers
  // from another function.
  func_inner(f, std::make_index_sequence<N>());
}

int main() {
  func<5>([](size_t i) {
    std::cout<< "Got index "<< i<< "\n";
  });
}

