#pragma once
#feature on forward
#include "tuple.hxx"

namespace algo {

// Transform
// (t, f) => (f(t_0),f(t_1),...,f(t_n))
template<typename F, typename Tup>
constexpr auto transform(F f, forward Tup tup : cir::tuple) {
  return cir::make_tuple(f(forward tup.[:])...);
}

template<typename F, typename Tup1, typename Tup2>
constexpr auto transform(F f, forward Tup1 tup1 : cir::tuple,
  forward Tup2 tup2 : cir::tuple) {
  return cir::make_tuple(f(forward tup1.[:], forward tup2.[:])...);
}

// Fold
// (t, v, f) => f(...f(f(v,t_0),t_1),...,t_n)
template<typename T, typename V, typename F>
constexpr auto fold(forward T tup : cir::tuple, forward V v, F f) {
  // Circle supports "functional folds."
  // Unary folds are only provided the pack.
  // Binary folds an initializer and the pack.
  // Binary operator fold repeats the operator:
  //   (init op ... op pack)
  // Binary functional folds states the function once:
  //   (init function ... pack)
  return (forward v f ... forward tup.[:]);
}

// Take
// Takes elements in the range [Begin, End)
template<size_t Begin, size_t End, typename T>
constexpr auto take(forward T tup : cir::tuple) {
  return cir::make_tuple(forward tup.[Begin:End] ...);
}

// Repeat
// Make a tuple with N instances of x.
template <size_t N, typename X>
constexpr auto repeat(const X& x) {
  return cir::make_tuple(for i : N => x);
}

// Concatenate
template<typename... Tuples>
constexpr cir::tuple<for typename Tuple : Tuples => Tuple~remove_cvref.[:] ...>
tuple_cat(forward Tuples... tpls : cir::tuple) {
  return { for tpl : forward tpls => tpl.[:] ... };
}

} // namespace cir