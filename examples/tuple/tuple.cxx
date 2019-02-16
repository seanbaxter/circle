#include "../include/serialize.hxx"

template<typename... types_t>
struct tuple_t {
  // For each type in the parameter pack...
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    // Declare a member named _i.
    types_t...[i] @(i); 
};

// tuple get() uses direct access to the member. @(i) is a dependent name
// (so tuple.@(i) is a dependent expression) during definition. It is 
// evaluated during substitution.
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

// Circle's make_tuple uses the standard decay protocol.
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
  // Use the aggregate initializer to construct the object.
  typedef tuple_t<special_decay_t<types_t>...> my_tuple;
  return my_tuple { std::forward<types_t>(args)... };
}

int main() {
  // Instantiate a tuple.
  tuple_t<int, std::string, double> my_tuple;

  // Initialize its members in three different ways.
  get<0>(my_tuple) = 1;           // Use std-style accessor.
  my_tuple._1 = "Second member";  // Use _1 member identifier.
  my_tuple.@(2) = 2.345;          // Use dynamic name.

  // Print the all the components of the tuple.
  stream(std::cout, my_tuple);
  std::cout<< std::endl;

  // Print the components of another tuple.
  using namespace std::literals;
  stream(std::cout, cir_tuple("A proper std::string"s, 1.618, 42));
  std::cout<< std::endl;

  return 0;
}