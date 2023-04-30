#ifdef USE_STD_TUPLE
#include <tuple>
namespace cir = std;
#include "algos-std.hxx"
#else
#include "tuple.hxx"
#include "algos-circle.hxx"
#endif
#include <functional>
#include <cstdio>

template<int... Is>
constexpr int test(std::integer_sequence<int, Is...>) {
  constexpr int Size = sizeof...(Is);

  // Fill a tuple with integers.
  cir::tuple tup1 = cir::make_tuple(Is...);

  // Fill with negative integers.
  cir::tuple tup2 = cir::make_tuple(-Is...);

  // Get the products.
  cir::tuple tup3 = algo::transform([](auto x, auto y) { return x * y; },
    tup1, tup2);

  // Take the 80% elements from the center.
  cir::tuple tup4 = algo::take<Size / 10, Size - Size / 10>(tup3);

  // Repeat 100 20% times.
  cir::tuple tup5 = algo::repeat<Size - std::tuple_size_v<decltype(tup4)>>(100);

  // Cat them together.
  cir::tuple tup6 = algo::tuple_cat(tup4, tup5);

  // We're back to Size elements.
  static_assert(std::tuple_size_v<decltype(tup6)> == Size);

  // Reduce the elements with a fold.
  int sum = algo::fold(tup6, 0, std::plus<int>());
  return sum;
}

// Make sure we don't actually generate LLVM code by doing a consteval and
// returning the result in a type. This can be called in an unevaluated context.
template<size_t N>
constexpr auto run_test() {
  constexpr int result = test(std::make_integer_sequence<int, N>());
  return std::integral_constant<int, result>();
}

template<int... Is>
int test_suite(std::integer_sequence<int, Is...>) {
  // Call run_test for each index in an unevaluated context. This is guaranteed
  // to only run the code in constexpr, so we aren't benchmarking backend
  // costs.
  int result = (... + decltype(run_test<Is>())::value);
  return result;
}

int main() {
  const int NumTests = BENCHMARK_SIZE;

  int x = test_suite(std::make_integer_sequence<int, NumTests>());
  printf("Result = %d\n", x);
}