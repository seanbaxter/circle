#pragma once
#include <utility>
#include <cstdlib>

template<typename... types_t>
struct tuple_t {
  // For each type in the parameter pack...
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    // Declare a member named _i.
    types_t...[i] @(i); 
};

template<size_t i, typename... types_t>
types_t...[i]& get(tuple_t<types_t...>& tuple) {
  return tuple.@(i);      // Allow accessing the member by name.
}

// Circle's version of tuple_element is implemented with a direct access also.
template<size_t i, typename type_t>
struct cir_tuple_element;

template<size_t i, typename... types_t>
struct cir_tuple_element<i, tuple_t<types_t...> > {
  typedef types_t...[i] type;
};

template<typename... types_t>
tuple_t<types_t...> cir_tuple(types_t&&... args) {

  // Alias templates to strip a wrapper.
  template <class T>
  struct unwrap_refwrapper {
    using type = T;
  };
   
  template <class T>
  struct unwrap_refwrapper<std::reference_wrapper<T>> {
    using type = T&;
  };
   
  // Alias template to strip references and decay types.
  template <class T>
  using special_decay_t = typename unwrap_refwrapper<
    typename std::decay<T>::type>::type;

  // Expand the type and function parameter packs to return a new tuple.
  typedef tuple_t<special_decay_t<types_t>...> this_tuple;
  return this_tuple { std::forward<types_t>(args)... };
}
