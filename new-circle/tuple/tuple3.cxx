#feature on forward
#include <tuple>
#include <iostream>

namespace cir {

// [[circle::native_tuple]] means .[] always accesses the data members
// directly, rather than going through the 'get' extension point.
template<typename... Ts>
struct [[circle::native_tuple]] tuple {
  // [[circle::no_unique_address_any]] aliases all empty types, distinct or
  // not, to the same address. This means any tuple with all empty elements
  // has size 1.
  [[circle::no_unique_address_any]] Ts ...m;
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

using T1 = std::integral_constant<size_t, 1>;
using T2 = std::integral_constant<size_t, 2>;

// Empty elements take up no space, even non-distinct ones.
using Tup = cir::tuple<T1, T2, T2, T1>;
static_assert(1 == sizeof(Tup));
static_assert(Tup~is_empty);

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
