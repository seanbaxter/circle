#pragma once
#include <utility>

using std::size_t;

namespace algo {

// Transform
// (t, f) => (f(t_0),f(t_1),...,f(t_n))

namespace detail {

template<typename F, typename Tup, size_t... Is>
constexpr auto transform(F f, Tup&& tup, std::index_sequence<Is...>) {
  return std::make_tuple(f(std::get<Is>(std::forward<Tup>(tup)))...);
}

template<typename F, typename Tup1, typename Tup2, size_t... Is>
constexpr auto transform(F f, Tup1&& tup1, Tup2&& tup2, std::index_sequence<Is...>) {
  return std::make_tuple(f(
    std::get<Is>(std::forward<Tup1>(tup1)), 
    std::get<Is>(std::forward<Tup2>(tup2))
  )...);
}

} // namespace detail

template<typename F, typename Tup>
constexpr auto transform(F f, Tup&& tup) {
  return detail::transform(
    f,
    std::forward<Tup>(tup),
    std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tup>>>()
  );
}

template<typename F, typename Tup1, typename Tup2>
constexpr auto transform(F f, Tup1&& tup1, Tup2&& tup2) {
  static_assert(
    std::tuple_size_v<std::remove_reference_t<Tup1>> == 
    std::tuple_size_v<std::remove_reference_t<Tup2>>);

  return detail::transform(
    f,
    std::forward<Tup1>(tup1),
    std::forward<Tup2>(tup2),
    std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tup1>>>()
  );
}

// Fold
// (t, v, f) => f(...f(f(v,t_0),t_1),...,t_n)

namespace detail {

// Overload for empty tuple.
template<typename Tup, typename V, typename F>
constexpr auto fold(Tup&& tup, V&& v, F f, std::index_sequence<>) {
  return std::forward<V>(v);
}

// Overload for non-empty tuple.
template<typename Tup, typename V, typename F, size_t I, size_t... Is>
constexpr auto fold(Tup&& tup, V&& v, F f, std::index_sequence<I, Is...>) {
  if constexpr(sizeof...(Is) == 0) {
    // Fold the remaining element with the accumulated value.
    return f(std::forward<V>(v), std::get<I>(std::forward<Tup>(tup)));

  } else {
    // Left-recurse.
    return fold(
      std::forward<Tup>(tup), 
      f(std::forward<V>(v), std::get<I>(std::forward<Tup>(tup))),
      f,
      std::index_sequence<Is...>()
    );
  }
}

} // namespace detail

template<typename Tup, typename V, typename F>
constexpr auto fold(Tup&& tup, V&& v, F f) {
  return detail::fold(
    std::forward<Tup>(tup), 
    std::forward<V>(v),
    f,
    std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tup>>>()
  );
}

// Take
// Takes elements in the range [Begin, End)

namespace detail {

template<size_t Begin, typename Tup, size_t... Is>
constexpr auto take(Tup&& tup, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Begin + Is>(std::forward<Tup>(tup))...);
}

} // namespace detail

template<size_t Begin, size_t End, typename Tup>
constexpr auto take(Tup&& tup) {
  return detail::take<Begin>(
    std::forward<Tup>(tup), 
    std::make_index_sequence<End - Begin>()
  );
}

// Repeat
// Make a tuple with N instances of x.

namespace detail {

template<typename X, size_t... Is>
constexpr auto repeat(const X& x, std::index_sequence<Is...>) {
  return std::make_tuple((void(Is), x)...);
}

} // namespace detail

template<size_t N, typename X>
constexpr auto repeat(const X& x) {
  return detail::repeat(x, std::make_index_sequence<N>());
}

// Concatenate
using std::tuple_cat;


} // namespace algo