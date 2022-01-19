#include <iostream>
#include <utility>

template<typename... Types>
struct variant {
  union {
    Types ...m;
  };
  uint8_t _index = 0;

  // Default initialize the 0th element.
  variant() : m...[0]() { }

  // Initialize the index indicated by I.
  template<size_t I, typename U>
  variant(std::in_place_index_t<I>, U&& u) : 
    m...[I](std::forward<U>(u)), _index(I) { }

  // Search for the index of the first type that matches T.
  template<typename T>
  static constexpr size_t index_of_type = T == Types ...?? int... : -1;

  // Count the number of types that match T.
  template<typename T>
  static constexpr size_t count_of_type = (0 + ... + (T == Types));

  // Initialize the type indicate by T.
  template<typename T, typename U, size_t I = index_of_type<T> >
  requires(1 == count_of_type<T>)
  variant(std::in_place_type_t<T>, U&& u) :
    m...[I](std::forward<U>(u)), _index(I) { }

  // Destroy the active variant member.
  ~variant() {
    _index == int... ...? m.~Types() : __builtin_unreachable();
  }
};

// Visit the active variant member.
template<typename F, typename... Types>
decltype(auto) visit(F f, variant<Types...>& var) {
  return var._index == int... ...? f(var. ...m) : __builtin_unreachable();
}

int main() {
  using Var = variant<int, double, std::string>;
  auto print_element = [](auto x) {
    std::cout<< decltype(x).string<< ": "<< x<< "\n";
  };

  // Default initialize element 0 (int 0).
  Var v1;
  visit(print_element, v1);

  // Initialize element 1 (double 6.67e-11)
  Var v2(std::in_place_index<1>, 6.67e-11);
  visit(print_element, v2);

  // Initialize the std::string element.
  Var v3(std::in_place_type<std::string>, "Hello variant");
  visit(print_element, v3);
}