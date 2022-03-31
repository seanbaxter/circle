#pragma once

#if !defined(__circle_build__) || __circle_build__ < 152
  #error Must compile with Circle build 165 or later
#endif

#if __cplusplus < 202002L
  #error Compile with -std=c++20
#endif

#include <tuple>
#include <memory>
#include <functional>

namespace circle {

template<class... Types>
class tuple;

} // namespace circle

namespace std {
  //////////////////////////////////////////////////////////////////////////////
  // [tuple.helper]

  template<class... Types>
  struct tuple_size<circle::tuple<Types...> > :
    public integral_constant<size_t, sizeof...(Types)> { };

  template<size_t I, class... Types>
  struct tuple_element<I, circle::tuple<Types...>> {
    static_assert(I < sizeof...(Types));
    using type = Types...[I];
  };

  // [tuple.traits]
  template<class... Types, class Alloc>
  struct uses_allocator<circle::tuple<Types...>, Alloc> { };

} // namespace std

namespace circle {

////////////////////////////////////////////////////////////////////////////////
// [tuple.elem]

template<size_t I, class Tuple, class...Types>
constexpr auto&& get(Tuple&& t forward : tuple<Types...>) noexcept {
  static_assert(I < sizeof...(Types));
  return t.template _get<I>();
}

template<class T, class Tuple, class... Types>
constexpr auto&& get(Tuple&& t forward : tuple<Types...>) noexcept {
  static_assert(1 == (0 + ... + (T == Types)));
  constexpr size_t I = T == Types ...?? int... : -1;
  return t.template _get<I>();
}

////////////////////////////////////////////////////////////////////////////////
// [tuple.creation]

template<class... TTypes>
constexpr tuple<TTypes.unwrap_ref_decay...> make_tuple(TTypes&&... t forward) {
  return { t... };
}

template<class... TTypes>
constexpr tuple<TTypes&&...> forward_as_tuple(TTypes&&... t forward) noexcept {
  return { t... };
}

template<class... TTypes>
constexpr tuple<TTypes&...> tie(TTypes&... t) noexcept {
  return tuple<TTypes&...>(t...);
}

template<class... Tuples>
constexpr tuple<
  for typename Tuple : Tuples => 
    Tuple.remove_reference.tuple_elements...
>
tuple_cat(Tuples&&... tpls forward) {
  return { for tpl : tpls => tpl... };
}

////////////////////////////////////////////////////////////////////////////////
// [tuple.rel]

template<class... TTypes, class... UTypes>
constexpr bool operator==(const tuple<TTypes...>& t, 
  const tuple<UTypes...>& u) {

  constexpr size_t N = sizeof...(TTypes);
  static_assert(N == sizeof...(UTypes));

  static_assert(requires { (bool)(get<int...>(t) == get<int...>(u)); },
    "no valid == for " + TTypes.string + " and " + UTypes.string)...;

  return (... && (get<int...(N)>(t) == get<int...>(u)));
}

template<class... TTypes, class... UTypes>
constexpr auto operator<=>(const tuple<TTypes...>& t, 
  const tuple<UTypes...>& u) { 

  static_assert(sizeof...(TTypes) == sizeof...(UTypes));
  static_assert(requires { std::declval<TTypes>() <=> std::declval<UTypes>(); },
    "no valid <=> for " + TTypes.string + " and " + UTypes.string)...;

  using Result = std::common_comparison_category_t<
    decltype(std::declval<TTypes>() <=> std::declval<UTypes>())...
  >;

  @meta for(size_t i : sizeof...(TTypes))
    if(Result c = get<i>(t) <=> get<i>(u); c != 0)
      return c;

  return Result::equivalent;
}

////////////////////////////////////////////////////////////////////////////////
// [tuple.special]

template<class... Types>
requires((... && std::is_swappable_v<Types>))
constexpr void swap(tuple<Types...>& x, tuple<Types...>& y) 
noexcept(noexcept(x.swap(y))) {
  x.swap(y);
}

////////////////////////////////////////////////////////////////////////////////
// [tuple.apply]

template<class F, class Tuple>
constexpr decltype(auto) apply(F&& f forward, Tuple&& t forward) {
  return std::invoke(f, t...);
}

template<class T, class Tuple>
constexpr T make_from_tuple(Tuple&& t forward) {
  return T(t...);
}

template<class... Types>
class tuple {
  [[no_unique_address]] Types ...m;

  // A dummy function to test default copy-list-initialization.
  template<typename T>
  static void dummy(const T&);

public:
  //////////////////////////////////////////////////////////////////////////////
  // [tuple.cnstr]

  // Default ctor.
  // The 0-element tuple has a trivial constructor.
  constexpr tuple() requires(0 == sizeof...(Types)) = default;

  // The expression inside explicit evaluates to true if and only if Ti
  // is not copy-list-initializable from an empty list for at least one i.
  // This default constructor value-initializes its members.
  explicit(!(... && requires { dummy<Types>({}); }))
  constexpr tuple()
  requires(
    sizeof...(Types) && 
    (... && std::is_default_constructible_v<Types>)
  ) : m()... { }

  // Construct from elements.
  constexpr explicit(!(... && std::is_convertible_v<const Types&, Types>))
  tuple(const Types&... x)
  requires(sizeof...(Types) > 0 && (... && std::is_copy_constructible_v<Types>)) :
    m(x)... { }

  // Element conversion constructor.
  template<class... UTypes>
  requires(
    // Constraints
    sizeof...(Types) >= 1 && 
    (... && std::is_constructible_v<Types, UTypes&&>) &&
    (  // disambiguating constraint
      sizeof...(Types) == 1 ??
        UTypes...[0].remove_cvref != tuple :
      sizeof...(Types) <= 3 ??
        UTypes...[0].remove_cvref != std::allocator_arg_t ||
        Types...[0].remove_cvref == std::allocator_arg_t :
      true
    )
  )
  constexpr explicit((... && std::is_convertible_v<UTypes&&, Types>))
  tuple(UTypes&&... u forward) : m(u)... { }

  // Copy and move constructors are defaulted.
  constexpr tuple(const tuple& u) = default;
  constexpr tuple(tuple&& u) = default;

  // Conversion constructor from tuple.
  template<class T, class... UTypes>
  requires(
    (... && std::is_constructible_v<Types, __copy_cvref(T&&, UTypes)>) &&
    (
      sizeof...(Types) != 1 |||
      (
        !std::is_convertible_v<T&&, Types...[0]> &&
        !std::is_constructible_v<Types...[0], T&&> &&
        Types...[0] != UTypes...[0]
      )      
    )
  )
  constexpr explicit(!(... && std::is_convertible_v<__copy_cvref(T&&, UTypes), Types>))
  tuple(T&& u forward : tuple<UTypes...>) : m(get<int...>(u))... { }

  // Conversion constructor from std::pair.
  template<class T, class U1, class U2>
  requires(
    sizeof...(Types) == 2 &&
    std::is_constructible_v<Types...[0], __copy_cvref(T&&, U1)> &&
    std::is_constructible_v<Types...[1], __copy_cvref(T&&, U2)>
  )
  constexpr explicit(
    !std::is_convertible_v<__copy_cvref(T&&, U1), Types...[0]> || 
    !std::is_convertible_v<__copy_cvref(T&&, U2), Types...[1]>
  )
  tuple(T&& u forward : std::pair<U1, U2>) : m((get<int...>(u)))... { }

  //////////////////////////////////////////////////////////////////////////////
  // Allocator-aware constructors.

  // Allocator-aware default constructor.
  template<class Alloc>
  requires((... && std::is_default_constructible_v<Types>))
  constexpr explicit(!(... && requires { dummy<Types>({}); }))
  tuple(std::allocator_arg_t, const Alloc& a) :
    m(std::make_obj_using_allocator<Types>(a))... { }

  // Allocator-aware constructor from elements.
  template<class Alloc>
  requires(sizeof...(Types) > 0 && (... && std::is_copy_constructible_v<Types>))
  constexpr explicit(!(... && std::is_convertible_v<const Types&, Types>))
  tuple(std::allocator_arg_t, const Alloc& a, const Types&... x) :
    m(std::make_obj_using_allocator<Types>(a, x))... { }

  // Allocator-aware converting constructor from elements.
  template<class Alloc, class... UTypes>
  requires(
    sizeof...(Types) >= 1 &&
    (... && std::is_constructible_v<Types, UTypes&&>) &&
    (sizeof...(Types) != 1 ||| UTypes...[0].remove_cvref != tuple)
  )
  constexpr explicit((... && std::is_convertible_v<UTypes&&, Types>))
  tuple(std::allocator_arg_t, const Alloc& a, UTypes&&... x forward) :
    m(std::make_obj_using_allocator<Types>(a, x))... { }

  // Converting allocator-aware constructor from tuple.
  template<class Alloc, class T, class... UTypes>
  requires(
    (... && std::is_constructible_v<Types, __copy_cvref(T&&, UTypes)>) &&
    (
      sizeof...(Types) != 1 |||
      (
        !std::is_convertible_v<T&&, Types...[0]> &&
        !std::is_constructible_v<Types...[0], T&&> &&
        Types...[0] != UTypes...[0]
      )
    )
  )
  constexpr explicit(!(... && std::is_convertible_v<__copy_cvref(T&&, UTypes), Types>))
  tuple(std::allocator_arg_t, const Alloc& a, T&& u forward : tuple<UTypes...>) :
    m(std::make_obj_using_allocator<Types>(a, get<int...>(u)))... { }

  // Converting allocator-aware constructor from std::pair.
  template<class Alloc, class T, class U1, class U2>
  requires(
    sizeof...(Types) == 2 &&
    std::is_constructible_v<Types...[0], __copy_cvref(T&&, U1)> &&
    std::is_constructible_v<Types...[1], __copy_cvref(T&&, U2)>
  )
  constexpr explicit(
    !std::is_convertible_v<__copy_cvref(T&&, U1), Types...[0]> || 
    !std::is_convertible_v<__copy_cvref(T&&, U2), Types...[1]>
  ) 
  tuple(std::allocator_arg_t, const Alloc& a, T&& u forward : std::pair<U1, U2>) :
    m(std::make_obj_using_allocator<Types>(a, get<int...>(u)))... { }


  //////////////////////////////////////////////////////////////////////////////
  // [tuple.assign]

  // Declare defaulted assignment.
  constexpr tuple& operator=(const tuple&) = default;
  constexpr tuple&& operator=(tuple&&) = default;

  template<class... UTypes, typename T>
  requires((... && std::is_assignable_v<Types, __copy_cvref(T&&, UTypes)>))
  constexpr tuple& operator=(T&& u forward : tuple<UTypes...>) {
    m = get<int...>(u) ...;
    return *this;
  }

  template<class U1, class U2, typename T>
  requires(
    sizeof...(Types) == 2 &&
    std::is_assignable_v<Types...[0]&, __copy_cvref(T&&, U1)> &&
    std::is_assignable_v<Types...[1]&, __copy_cvref(T&&, U2)>
  )
  constexpr tuple& operator=(T&& u forward : std::pair<U1, U2>) {
    m = get<int...>(u) ...;
    return *this;
  }

  // Give the internal function the name _get so it doesn't break ADL for
  // getters of other types.
  template<size_t I, class Self>
  auto&& _get(this Self&& self : tuple) noexcept {
    return self. ...m...[I];
  }

  // [tuple.swap]
  constexpr void swap(tuple& rhs) 
  noexcept((... && std::is_nothrow_swappable_v<Types>)) {
    using std::swap;
    swap(m, rhs. ...m)...;
  }

  constexpr void swap(const tuple& rhs) 
  noexcept((... && std::is_nothrow_swappable_v<const Types>)) {
    using std::swap;
    swap(m, rhs. ...m)...;
  }
};

// Deduction guides for std::tuple
template<class... UTypes>
tuple(UTypes...) -> tuple<UTypes...>;

template<class T1, class T2>
tuple(std::pair<T1, T2>) -> tuple<T1, T2>;

template<class Alloc, class... UTypes>
tuple(std::allocator_arg_t, Alloc, UTypes...) -> tuple<UTypes...>;

template<class Alloc, class T1, class T2>
tuple(std::allocator_arg_t, Alloc, std::pair<T1, T2>) -> tuple<T1, T2>;

template<class Alloc, class... UTypes>
tuple(std::allocator_arg_t, Alloc, tuple<UTypes...>) -> tuple<UTypes...>;

} // namespace circle
