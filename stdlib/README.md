# Circle implementations of Standard Library classes.

## Contents.

1. [Member pack declarations](#member-pack-declarations)
    * [Basic tuple](#basic-tuple)
    * [Basic variant](#basic-variant)
    * [Basic mdspan](#basic-mdspan)
2. [Deduced forwarding references](#deduced-forwarding-references)
3. [Circle Imperative Arguments](#circle-imperative-arguments)
    * [Create an n-length set](#create-an-n-length-set)
    * [Tuple cat](#tuple-cat)

## 1. Member pack declarations.

Declare a pack of non-static data members with the member pack declaration syntax. Use `...` before the _declarator-id_, as if you were writing a function or template parameter pack. This is compatible with the description in [P1858R2 - Generalized pack declaration and usage](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1858r2.html#member-packs). Use a member pack in a struct to define a [tuple-like thing](#basic-tuple). Use a member pack in a union to define a [variant-like thing](#basic-variant). Use a member pack and partially-static storage templates to define the extents in an [mdspan-like thing](#basic-mdspan).

### Basic tuple.

[**tuple1.cxx**](tuple1.cxx) - [Compiler Explorer](https://godbolt.org/z/5srsvdTqb)
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
  std::cout<< "  "<< decltype(A).member_decl_strings<< ": "<< A.m<< "\n" ...;

  // It even works with CTAD! Deduced through the parameter pack.
  tuple B {
    6ll, 1.618f, true
  };
  std::cout<< "B:\n";
  std::cout<< "  "<< decltype(B).member_decl_strings<< ": "<< B.m<< "\n" ...;
}
```
```
$ circle tuple1.cxx && ./tuple1
A:
  int m0: 5
  double m1: 3.14
  char m2: X
B:
  long long m0: 6
  float m1: 1.618
  bool m2: 1
```

The basic tuple becomes a one-liner. Access the data members of objects by naming the member pack. This yields a pack expression, which must be expanded with `...`. As a bonus, class template argument deduction even works through member pack declarations. We attempt aggregate initialization of `B` with a `long long`, `float` and `bool`, and the class template is indeed specialized with those arguments.

When the member pack is instantiated, concrete member declarations get names with the ordinal of the expansion postfixed to the pack's name. Eg, `m` expands to members `m0`, `m1`, `m2` and so on. These are useful for the purpose of reflection. Member traits like `member_names` and `member_decl_strings` return member names and declarations as constant character arrays, and use these substituted names.

### Basic variant.

[**variant1.cxx**](variant1.cxx) - [Compiler Explorer](https://godbolt.org/z/shYG4v91P)
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
```
```
$ circle variant1.cxx && ./variant1
int: 0
double: 6.67e-11
std::basic_string<char, std::char_traits<char>, std::allocator<char>>: Hello variant
```

To write a basic variant, make a member pack declaration inside an unnamed union. The [pack subscript operator](https://github.com/seanbaxter/circle/tree/master/universal#pack-subscripts-and-slices) `...[I]` provides direct access to specific instantiated data members.  Use it inside the _mem-initializer-list_ to name a member to initialize.

The [multi-conditional operator](https://github.com/seanbaxter/circle/tree/master/conditional#multi-conditional---) `...?:` makes it easy to generate a cascade of ternary operators, which serve as a higher-level switch to bridge runtime values (like `_index)` with compile-time concerns (like a particular data member). The pseudo-destructor call in the variant's destructor, or the callable invocation in `visit` are both satisfied in one line with this operator.

### Basic mdspan.

[**extent1.cxx**](extent1.cxx)
```cpp
#include <type_traits>
#include <limits>
#include <iostream>

constexpr size_t dynamic_extent = size_t.max;

template<typename Type>
concept SizeType = std::is_convertible_v<Type, size_t>;

template<size_t index, size_t Extent>
struct _storage_t {
  // static storage.
  constexpr _storage_t(size_t extent) noexcept { }
  static constexpr size_t extent = Extent;
};

template<size_t index>
struct _storage_t<index, dynamic_extent> {
  // dynamic storage.
  constexpr _storage_t(size_t extent) noexcept : extent(extent) { }
  size_t extent;
};

template<size_t... Extents>
struct extents {
  // Partial static storage.
  [[no_unique_address]] _storage_t<int..., Extents> ...m;

  // Count the rank (number of Extents).
  static constexpr size_t rank() noexcept {
    return sizeof... Extents;
  }

  // Count the dynamic rank (number of Extents equal to dynamic_equal).
  static constexpr size_t rank_dynamic() noexcept {
    return (0 + ... + (dynamic_extent == Extents));
  }

  // Dynamic access to extents.
  constexpr size_t extent(size_t i) const noexcept {
    return i == int... ...? m.extent : 0;
  }

  // Construct from one index per extent.
  template<SizeType... IndexTypes>
  requires(sizeof...(IndexTypes) == rank())
  constexpr extents(IndexTypes... exts) noexcept : m(exts)... { }

  // Map index I to index of dynamic extent J.
  template<size_t I>
  static constexpr size_t find_dynamic_index =
    (0 + ... + (dynamic_extent == Extents...[:I]));

  // Construct from one index per *dynamic extent*.
  template<SizeType... IndexTypes>
  requires(
    sizeof...(IndexTypes) != rank() && 
    sizeof...(IndexTypes) == rank_dynamic()
  )
  constexpr extents(IndexTypes... exts) noexcept : m(
    dynamic_extent == Extents ??
      exts...[find_dynamic_index<int...>] :
      Extents
  )... { }
};

int main() {
  using Extents = extents<3, 4, dynamic_extent, dynamic_extent, 7>;

  // Initialize extents with one value per extent.
  Extents e1(3, 4, 5, 6, 7);

  // Initialize extents with one value per *dynamic extent*. The static extents
  // are inherited from the template arguments.
  Extents e2(5, 6);

  for(int i : Extents::rank())
    std::cout<< i<< ": "<< e1.extent(i)<< " - "<< e2.extent(i)<< "\n";
}
```

## Deduced forwarding references.



## Circle Imperative Arguments.

### Create an n-length set.

### Tuple cat.

[libstdc++ implementation](https://github.com/gcc-mirror/gcc/blob/7adcbafe45f8001b698967defe682687b52c0007/libstdc%2B%2B-v3/include/std/tuple#L1693)

[**tuple_cat.cxx**](tuple_cat.cxx)
```cpp
#include <tuple>
#include <string>
#include <iostream>

template<class... Tuples>
constexpr std::tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat1(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      auto N : Ti.remove_reference.tuple_size =>
        get<int...(N)>(std::forward<Ti>(tpls...[i]))...
  };
}

template<class... Tuples>
constexpr std::tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat2(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      std::forward<Ti>(tpls...[i]) ...
  };
}

int main() {
  using namespace std::string_literals;
  auto t1 = std::make_tuple(1, 2.2, "Three");
  auto t2 = std::make_tuple("Four"s, 5i16);
  auto t3 = std::make_tuple(6.6f, 7ull);

  auto cat  = std::tuple_cat(t1, t2, t3);
  auto cat1 = tuple_cat1(t1, t2, t3);
  auto cat2 = tuple_cat2(t1, t2, t3);
  
  std::cout<< "cat == cat1 is "<< (cat == cat1 ? "true\n" : "false\n");
  std::cout<< "cat == cat2 is "<< (cat == cat2 ? "true\n" : "false\n");

  std::cout<< decltype(cat2).tuple_elements.string<< ": "<< cat2.[:]<< "\n" ...;  
}
```

