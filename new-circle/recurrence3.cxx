#feature on forward tuple recurrence
#include <tuple>
#include <iostream>

// Rank 0 case. This just returns the parameter.
template <typename Tuple>
constexpr decltype(auto) getN(forward Tuple t) noexcept {
  return forward t;
}

// Rank1 or higher case.
template<size_t I0, size_t... Is, typename Tuple>
constexpr decltype(auto) getN(forward Tuple tuple) noexcept {
  // Recursive on getN. Exits out with the overload above.
  return getN<Is...>(get<I0>(forward tuple));
}

// [recurrence] implementation:
template<size_t... Is, typename Tuple>
constexpr decltype(auto) getN2(forward Tuple tuple) noexcept {
  // Non-recursive. This does not call back into getN2.
  return (get<Is>(recurrence forward tuple) ...);
}

int main() {
  // Use [tuple] to declare a complicated tuple.
  auto tup = (1, "Hello", (2i16, (3.14, ("Recurrence", "World"), 4ul)), 5i8);

  // We want to pull out the "Recurrence" element, which has index:
  // [2, 1, 1, 0]

  // We can use the old recursive approach:
  std::cout<< getN<2, 1, 1, 0>(tup)<< "\n";

  // Or the new recurrence approach:
  std::cout<< getN2<2, 1, 1, 0>(tup)<< "\n";
}
