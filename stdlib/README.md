# Circle implementations of Standard Library class.

## Contents.

1. [Member pack declarations](#member-pack-declarations)

## 1. Member pack declarations.

Declare a pack of non-static data members with the member pack declaration syntax. Use `...` before the _declarator-id_, as if you were writing a function or template parameter pack. This is compatible with the description in [P1858R2 - Generalized pack declaration and usage](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1858r2.html#member-packs).

### Basic tuple

[**tuple1.cxx**](tuple1.cxx) - [Compiler Explorer](https://godbolt.org/z/j8ddPP9aK)
```cpp
#include <iostream>

template<typename... Types>
struct tuple {
  [[no_unique_address]] Types ...m;
};

int main() {
  // Declare and use the aggregate initializer.
  tuple<int, double, char> A {
    5, 3.14, 'X'
  };
  std::cout<< "A:\n";
  std::cout<< "  "<< decltype(A).member_type_strings<< ": "<< A.m<< "\n" ...;

  // It even works with CTAD! Deduced through the parameter pack.
  tuple B {
    6ll, 1.618f, true
  };
  std::cout<< "B:\n";
  std::cout<< "  "<< decltype(B).member_type_strings<< ": "<< B.m<< "\n" ...;
}
```
```
$ circle tuple1.cxx && ./tuple1
A:
  int: 5
  double: 3.14
  char: X
B:
  long long: 6
  float: 1.618
  bool: 1
```

The basic tuple becomes a one-liner. Access the data members of objects by naming the member pack. This yields a pack expression, which must be expanded with `...`. As a bonus, class template argument deduction even works through member pack declarations. We attempt aggregate initialization of `B` with a `long long`, `float` and `bool`, and the class template is indeed specialized with those arguments.

### Basic variant

[**variant1.cxx**](variant1.cxx) - [Compiler Explorer](https://godbolt.org/z/jK4fvGM1v)
```cpp
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
  static constexpr size_t count_of_type = (... + (T == Types));

  // Initialize the type indicate by T.
  template<typename T, typename U, size_t I = index_of_type<T> >
  requires(1 == count_of_type<T>)
  variant(std::in_place_type_t<T>, U&& u) :
    m...[I](std::forward<U>(u)), _index(I) { }

  // Destroy the active variant member.
  ~variant() {
    _index == int... ...? m.~Types() : __builtin_unreachable();
  }

  // Use a constrained forward reference deduced this to implement all
  // get cv-ref combinations.
  template<size_t I, typename Self>
  auto&& get(this Self&& self : variant) {
    return self. ...m...[I];
  } 
};

// Visit the active variant member.
template<typename F, typename... Types>
decltype(auto) visit(F f, variant<Types...>& var) {
  constexpr size_t N = sizeof...(Types);
  var._index == int...(N) ...? 
    f(var.template get<int...>()) :
    __builtin_unreachable();
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
```
```
$ circle variant1.cxx && ./variant1
int: 0
double: 6.67e-11
std::basic_string<char, std::char_traits<char>, std::allocator<char>>: Hello variant
```

To write a basic variant, make a member pack declaration inside an unnamed union. The [pack subscript operator](https://github.com/seanbaxter/circle/tree/master/universal#pack-subscripts-and-slices) `...[I]` provides direct access to specific instantiated data members.  Use it inside the _mem-initializer-list_ to name a member to initialize.

The [multi-conditional operator](https://github.com/seanbaxter/circle/tree/master/conditional#multi-conditional---) `...?:` makes it easy to generate a cascade of ternary operators, which serve as a higher-level switch to bridge runtime values (like `_index)` with compile-time concerns (like a particular data member). The pseudo-destructor call in the variant's destructor, or the callable invocation in `visit` are both satisfied in one line with this operator.

## Basic mdspan

