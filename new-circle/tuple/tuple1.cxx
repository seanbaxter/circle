#feature on forward
#include <tuple>
#include <iostream>

namespace cir {

template<typename... Ts>
struct tuple {
  [[no_unique_address]] Ts ...m;   // Wow!
};

// Provide a getter for existing code that expects this extension point.
template<size_t I, typename Tup>
auto&& get(forward Tup tup : tuple) noexcept {
  return forward tup. ...m ...[I];
}

// Be careful when using make_tuple. ADL may pull in std::make_tuple
// if you pass any argument types from that namespace. Please qualify
// your function calls.
template<typename... Ts>
tuple<Ts~remove_cvref...> make_tuple(forward Ts... x) {
  return { forward x ... };
}

template<typename F, typename Tup>
auto transform(F f, forward Tup tup : tuple) {
  return cir::make_tuple(f(forward tup.[:])...);
}

} // namespace cir

namespace std {

// Implement the C++ extension points.
template<size_t I, typename... Ts>
struct tuple_element<I, cir::tuple<Ts...>> {
  using type = Ts...[I];
};

template<typename... Ts>
struct tuple_size<cir::tuple<Ts...>> : 
  integral_constant<size_t, sizeof...(Ts)> { };

} // namespace std

int main() {
  // Declare a big tuple and fill it with ascending numbers.
  const size_t N = 50;
  auto tup1 = cir::make_tuple(int...(N)...);

  // sqrt the elements. Promote to double.
  auto tup2 = cir::transform([](auto x) { return sqrt(x); }, tup1);

  // Write the elements to the terminal. 
  std::cout<< "{:3}: ".format(int...)<< tup2.[:]<< "\n" ...;
}
