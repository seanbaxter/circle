#pragma once

#include <tuple>
#include <memory>

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
auto&& get(Tuple&& t : tuple<Types...>) noexcept {
  static_assert(I < sizeof...(Types));
  return std::forward<Tuple>(t).template _get<I>();
}

template<class T, class Tuple, class... Types>
auto&& get(Tuple&& t : tuple<Types...>) noexcept {
  // Mandates: The Type T occures exactly once in types.
  static_assert(1 == (... + (T == Types)));

  constexpr size_t I = T == Types ...?? int... : -1;
  return std::forward<Tuple>(t).template _get<I>();
}


////////////////////////////////////////////////////////////////////////////////
// [tuple.creation]

template<class... TTypes>
constexpr tuple<TTypes.unwrap_ref_decay...> make_tuple(TTypes&&... t) {
  return { std::forward<TTypes>(t)... };
}

template<class... TTypes>
constexpr tuple<TTypes&&...> forward_as_tuple(TTypes&&... t) noexcept {
  return { std::forward<TTypes>(t)... };
}

template<class... TTypes>
constexpr tuple<TTypes&...> tie(TTypes&... t) noexcept {
  return tuple<TTypes&...>(t...);
}

template<class... Tuples>
constexpr tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      auto N : Ti.remove_reference.tuple_size =>
        get<int...(N)>(std::forward<Ti>(tpls...[i]))...
  };
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

  constexpr size_t N = sizeof...(TTypes);
  static_assert(N == sizeof...(UTypes));

  static_assert(requires { std::declval<TTypes>() <=> std::declval<UTypes>(); },
    "no valid <=> for " + TTypes.string + " and " + UTypes.string)...;

  using Result = std::common_comparison_category_t<
    decltype(std::declval<TTypes>() <=> std::declval<UTypes>())...
  >;

  @meta for(size_t i : N)
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
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  constexpr size_t N = Tuple.remove_reference.tuple_size;
  return std::forward<F>(f)(get<int...(N)>(std::forward<Tuple>(t))...);
}

template<class T, class Tuple>
constexpr T make_from_tuple(Tuple&& t) {
  constexpr size_t N = Tuple.remove_reference.tuple_size;
  return T(get<int...(N)>(std::forward<Tuple>(t))...);
}


template<class... Types>
class tuple {
  template<int I, typename T>
  struct storage_t { T x; };
  [[no_unique_address]] storage_t<int..., Types> ...m;

  // A dummy function to test default copy-list-initialization.
  template<typename T>
  void dummy(const T&);

public:
  //////////////////////////////////////////////////////////////////////////////
  // [tuple.cnstr]

  // Default ctor.
  // The expression inside explicit evaluates to true if and only if Ti
  // is not copy-list-initializable from an empty list for at least one i.
  explicit(!(... && requires { dummy<Types>({}); }))
  constexpr tuple()
  requires((... && std::is_default_constructible_v<Types>)) = default;

  // Construct from elements.
  constexpr explicit(!(... && std::is_convertible_v<const Types&, Types>))
  tuple(const Types&... x)
  requires(sizeof...(Types) > 0 && (... && std::is_copy_constructible_v<Types>)) :
    m { x }... { }

  // Element conversion constructor.
  template<class... UTypes>
  requires(
    // Constraints
    sizeof...(Types) >= 1 && 
    (... && std::is_constructible_v<Types, UTypes>) &&
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
  tuple(UTypes&&... u) :
    m { std::forward<UTypes>(u) }... { }

  // Copy and move constructors are defaulted.
  constexpr tuple(const tuple& u) = default;
  constexpr tuple(tuple&& u) = default;

  // Conversion constructor from tuple.
  template<class T, class... UTypes>
  requires(
    (... && std::is_constructible_v<Types, UTypes&&>) &&
    (
      sizeof...(Types) != 1 |||
      (
        !std::is_convertible_v<T&&, Types...[0]> &&
        !std::is_constructible_v<Types...[0], T&&> &&
        Types...[0] != UTypes...[0]
      )      
    )
  )
  constexpr explicit(!(... && std::is_convertible_v<UTypes&&, Types>))
  tuple(T&& u : tuple<UTypes...>) :
    m { get<int...>(std::forward<T>(u)) }... { }

  // Conversion constructor from std::pair.
  template<class T, class U1, class U2>
  requires(
    sizeof...(Types) == 2 &&
    std::is_constructible_v<Types...[0], U1&&> &&
    std::is_constructible_v<Types...[1], U2&&>
  )
  constexpr explicit(
    !std::is_convertible_v<U1&&, Types...[0]> || 
    !std::is_convertible_v<U2&&, Types...[1]>
  )
  tuple(T&& u : std::pair<U1, U2>) :
    m { (get<int...>(std::forward<T>(u))) }... { }

  //////////////////////////////////////////////////////////////////////////////
  // Allocator-aware constructors.

  // Allocator-aware default constructor.
  template<class Alloc>
  requires((... && std::is_default_constructible_v<Types>))
  constexpr explicit(!(... && requires { dummy<Types>({}); }))
  tuple(std::allocator_arg_t, const Alloc& a) :
    m { std::make_obj_using_allocator<Types>(a) }... { }

  // Allocator-aware constructor from elements.
  template<class Alloc>
  requires(sizeof...(Types) > 0 && (... && std::is_copy_constructible_v<Types>))
  constexpr explicit(!(... && std::is_convertible_v<const Types&, Types>))
  tuple(std::allocator_arg_t, const Alloc& a, const Types&... x) :
    m { std::make_obj_using_allocator<Types>(a, x) }... { }

  // Allocator-aware converting constructor from elements.
  template<class Alloc, class... UTypes>
  requires(
    sizeof...(Types) >= 1 &&
    (... && std::is_constructible_v<Types, UTypes&&>) &&
    (sizeof...(Types) != 1 ||| UTypes...[0].remove_cvref != tuple)
  )
  constexpr explicit((... && std::is_convertible_v<UTypes&&, Types>))
  tuple(std::allocator_arg_t, const Alloc& a, UTypes&&... x) :
    m { std::make_obj_using_allocator<Types>(a, std::forward<UTypes>(x)) }... { }

  // Converting allocator-aware constructor from tuple.
  template<class Alloc, class T, class... UTypes>
  requires(
    (... && std::is_constructible_v<Types, UTypes&&>) &&
    (
      sizeof...(Types) != 1 |||
      (
        !std::is_convertible_v<T&&, Types...[0]> &&
        !std::is_constructible_v<Types...[0], T&&> &&
        Types...[0] != UTypes...[0]
      )
    )
  )
  constexpr explicit(!(... && std::is_convertible_v<UTypes&&, Types>))
  tuple(std::allocator_arg_t, const Alloc& a, T&& u : tuple<UTypes...>) :
    m { std::make_obj_using_allocator<Types>(a, get<int...>(std::forward<T>(u))) }... { }

  // Converting allocator-aware constructor from std::pair.
  template<class Alloc, class T, class U1, class U2>
  requires(
    sizeof...(Types) == 2 &&
    std::is_constructible_v<Types...[0], U1&&> &&
    std::is_constructible_v<Types...[1], U2&&>
  )
  constexpr explicit(
    !std::is_convertible_v<U1&&, Types...[0]> || 
    !std::is_convertible_v<U2&&, Types...[1]>
  ) 
  tuple(std::allocator_arg_t, const Alloc& a, T&& u : std::pair<U1, U2>) :
    m{ std::make_obj_using_allocator<Types>(a, get<int...>(std::forward<T>(u))) } { }


  //////////////////////////////////////////////////////////////////////////////
  // [tuple.assign]

  // Declare defaulted assignment.
  constexpr tuple& operator=(const tuple&) = default;
  constexpr tuple&& operator=(tuple&&) = default;

  template<class... UTypes, typename T>
  requires((... && std::is_assignable_v<Types, const UTypes&>))
  constexpr tuple& operator=(const tuple<UTypes...>& u) {
    m.x = get<int...>(u) ...;
    return *this;
  }

  template<class U1, class U2>
  requires(
    sizeof...(Types) == 2 &&
    std::is_assignable_v<Types...[0]&, const U1&> &&
    std::is_assignable_v<Types...[1]&, const U2&>
  )
  constexpr tuple& operator=(const std::pair<U1, U2>& u) {
    m.x = get<int...>(u) ...;
    return *this;
  }

  // Give the internal function the name _get so it doesn't break ADL for
  // getters of other types.
  template<size_t I, class Self>
  auto&& _get(this Self&& self : tuple) noexcept {
    return self. ...m...[I].x;
  }

  // [tuple.swap]
  constexpr void swap(tuple& rhs) 
  noexcept((... && std::is_nothrow_swappable_v<Types>)) {
    using std::swap;
    swap(m.x, rhs. ...m.x)...;
  }

  constexpr void swap(const tuple& rhs) 
  noexcept((... && std::is_nothrow_swappable_v<const Types>)) {
    using std::swap;
    swap(m.x, rhs. ...m.x)...;
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
