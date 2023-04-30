#pragma once
#feature on forward
#include <tuple>

using std::size_t;

namespace cir {

#if defined(USE_EASY_TUPLE)

template<typename... Ts>
struct tuple {
  [[no_unique_address]] Ts ...m;   // Wow!
};

template<size_t I, typename Tup>
constexpr auto&& get(forward Tup tup : tuple) noexcept {
  return forward tup. ...m ...[I];
}

#elif defined(USE_HARD_TUPLE)

namespace detail {

template<size_t I, typename T, 
  bool is_empty = T~is_empty && T~is_default_constructible>
struct EBO {
  // Don't inherit from the base class, because that would prevent the 
  // empty base optimization from aliasing elements of the same type.
  constexpr T get_m() const  { return { }; }
};


template<size_t I, typename T>
struct EBO<I, T, false> {
  constexpr T& get_m() & noexcept { return m; }
  constexpr const T& get_m() const & noexcept { return m; }
  constexpr T&& get_m() && noexcept { return m; }

  // Make the non-empty (or the final) type a data memebr.
  T m;
};

} // namespace detail

template<typename... Ts>
struct tuple {
  [[no_unique_address]] detail::EBO<int..., Ts> ...m;
};

template<size_t I, typename Tup>
constexpr auto&& get(forward Tup tup : tuple) noexcept {
  return forward tup. ...m ...[I].get_m();
}

#elif defined(USE_CIRCLE_TUPLE)

template<typename... Ts>
struct [[circle::native_tuple]] tuple {
  [[circle::no_unique_address_any]] Ts ...m;
};

template<size_t I, typename Tup>
constexpr auto&& get(forward Tup tup : tuple) {
  // Call the getter.
  return forward tup. ...m ...[I].get_m();
}

#else
#error expected USE_STD_TUPLE, USE_EASY_TUPLE, USE_HARD_TUPLE or USE_CIRCLE_TUPLE
#endif

// Be careful when using make_tuple. ADL may pull in std::make_tuple
// if you pass any argument types from that namespace. Please qualify
// your function calls.
template<typename... Ts>
constexpr tuple<Ts~remove_cvref...> make_tuple(forward Ts... x) {
  return { forward x ... };
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
