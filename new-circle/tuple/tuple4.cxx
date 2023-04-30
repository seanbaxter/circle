#include <tuple>
#include <iostream>

// Duplicate all elements of a tuple.
template<typename... Ts>
auto duplicate_elements(const std::tuple<Ts...>& tup) {
  return std::make_tuple(.{tup.[:], tup.[:]}...);
}

int main() {
  auto tup1 = std::make_tuple(1, 2.2, "A tuple");
  auto tup2 = duplicate_elements(tup1);

  std::cout<< decltype(tup2).[:]~string<< " : "<< tup2.[:]<< "\n" ...;
}