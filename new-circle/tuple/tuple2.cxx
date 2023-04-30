#feature on forward
#include <tuple>
#include <iostream>

namespace cir {

namespace detail {

template<size_t I, typename T, 
  bool is_empty = T~is_empty && T~is_default_constructible>
struct EBO {
  // Don't inherit from the base class, because that would prevent the 
  // empty base optimization from aliasing elements of the same type.
  T get_m() const { return { }; }
};


template<size_t I, typename T>
struct EBO<I, T, false> {
  T& get_m() & noexcept { return m; }
  const T& get_m() const & noexcept { return m; }
  T&& get_m() && noexcept { return m; }

  // Make the non-empty (or the final) type a data memebr.
  T m;
};

} // namespace detail

template<typename... Ts>
struct tuple {
  // We really should have constructors to ease initialization of the
  // EBO members.

  // Use int... to set each EBO to a different index.
  [[no_unique_address]] detail::EBO<int..., Ts> ...m;
};

template<size_t I, typename Tup>
auto&& get(forward Tup tup : tuple) {
  // Call the getter.
  return forward tup. ...m ...[I].get_m();
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

// Prove that we can alias multiple non-distinct empty types.
using T1 = std::integral_constant<size_t, 1>;
using T2 = std::integral_constant<size_t, 2>;

// Empty elements take up no space, even non-distinct ones.
using Tup = cir::tuple<T1, T2, T2, T1>;
static_assert(1 == sizeof(Tup));
static_assert(Tup~is_empty);

// With std::tuple, empty elements may not alias with others of the same type.
// This is a two byte tuple. T1 and T2 are at address 0. T2 and T1 are at
// address 1.
using StdTup = std::tuple<T1, T2, T2, T1>;
static_assert(2 == sizeof(StdTup));
static_assert(StdTup~is_empty);

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
