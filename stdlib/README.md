# Circle implementations of Standard Library classes.

This page highlights Circle language features that aided in the Circle rewrite of three C++ Standard Library classes:

* [std::tuple](https://eel.is/c++draft/tuple) - 350 lines! [Implementation](../tuple/tuple.hxx) - [Notes](../tuple#circle-tuple)
* [std::variant](https://eel.is/c++draft/variant) - 650 lines! [Implementation](../variant/variant.hxx) - [Notes](../tvariantcircle-variant)
* [std::mdspan](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0009r14.html) - 700 lines! [Implementation](https://github.com/seanbaxter/mdspan/blob/circle/circle/experimental/mdspan.hpp) - [Notes](https://github.com/seanbaxter/mdspan#mdspan-circle)

The implementations draw on lots of features unique to Circle, some even motivated by these STL components.

### Features unique to Circle/C++:

* [Member pack declarations](https://github.com/seanbaxter/circle/tree/master/tuple#data-member-packs) `...m`
* [Pack subscripts and slices](https://github.com/seanbaxter/circle/tree/master/universal#pack-subscripts-and-slices) `...[I]` and `...[begin:end:step]`
* [Tuple subscripts and slices](https://github.com/seanbaxter/circle/tree/master/universal#tuple-subscripts-and-slices) `.[I]` and `.[begin:end:step]`
* [Pack indices](https://github.com/seanbaxter/circle/tree/master/universal#pack-indices) `int...`, `int...(N)` and `int...(begin:end:step)`
* [Deduced forwarding references](https://github.com/seanbaxter/circle/tree/master/tuple#deduced-forward-references)
* Enhanced conditional operators [`??:`](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-conditional--), [`...?:`](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#multi-conditional---) and [`...??:`](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-multi-conditional---)
* [Member type traits](https://github.com/seanbaxter/circle/tree/master/imperative#type-traits)
* [Generic comparisons](https://github.com/seanbaxter/circle/tree/master/imperative#is_specialization)
* [_argument-for_](https://github.com/seanbaxter/circle/tree/master/imperative#argument-for)
* [`__visit`](https://github.com/seanbaxter/circle/tree/master/variant#visit) compiler builtin
* [`__preferred_copy_init`](https://github.com/seanbaxter/circle/tree/master/variant#converting-constructor) and [`__preferred_assignment`](https://github.com/seanbaxter/circle/tree/master/variant#converting-assignment) compiler builtins
* [Pack `static_assert`](https://github.com/seanbaxter/circle/tree/master/variant#comparison-and-relational-operators)

## Contents.

1. [Member pack declarations](#1-member-pack-declarations)
    * [Basic tuple](#basic-tuple)
    * [Basic variant](#basic-variant)
    * [Basic mdspan](#basic-mdspan)
2. [Circle Imperative Arguments](#2-circle-imperative-arguments)
    * [Create an n-length set](#create-an-n-length-set)
    * [Tuple cat](#tuple-cat)
3. [Deduced forwarding references](#3-deduced-forwarding-references)
4. [N-dimensional Visit](#4-n-dimensional-visit)  
5. [Builtins for overload resolution](#5-builtins-for-overload-resolution)
6. [First-class tuple support](#6-first-class-tuple-support)

## 1. Member pack declarations.

Declare a pack of non-static data members with the member pack declaration syntax. Use `...` before the _declarator-id_, as if you were writing a function or template parameter pack. This is compatible with the description in [P1858R2 - Generalized pack declaration and usage](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1858r2.html#member-packs). Use a member pack in a struct to define a [tuple-like thing](#basic-tuple). Use a member pack in a union to define a [variant-like thing](#basic-variant). Use a member pack and partially-static storage templates to define the extents in an [mdspan-like thing](#basic-mdspan).

### Basic tuple.

[**tuple1.cxx**](tuple1.cxx) - [Compiler Explorer](https://godbolt.org/z/nMb4jz7d8)
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

Note that dependent name lookup of a member pack requires a disambiguation token `...` just before the member name. 

```cpp
template<typename F, typename... Types>
decltype(auto) visit(F f, variant<Types...>& var) {
  return var._index == int... ...? f(var. ...m) : __builtin_unreachable();
}
```

The program is ill-formed if, during substitution, the compiler finds a member pack without a preceding `...` token, or if it finds a non-member pack with the token.

### Basic mdspan.

[mdspan](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0009r14.html) is a multi-dimensional span class, that can be specialized over any mixture of static and dynamic extents. Static extents take no storage, and are pinned to template parameters. The challenge of implementation is ensuring _partially-static storage_, so that only the dynamic extents take storage, while providing both dynamic and static access to this now-irregular collection of extents.

```cpp
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
};
```

The heart of the Circle mdspan [implementation](https://github.com/seanbaxter/mdspan#mdspan-circle) is a `[[no_unique_address]]` member pack declaration of `storage_t` types. When the `Extent` template parameter equals `dynamic_extent`, then the extent is stored by the non-static data member `extent`. Otherwise, the extent is indicated by the static data member of the same name. `_storage_t` classes specialized on static extents are _empty classes_, and we arrange that they take no space in the `extents` class by using the `[[no_unique_address]]` attribute and by making their types unique, specializing the class templates with the index of the extent within its collection. Empty unique types marked `[[no_unique_address]]` alias to the same layout offset, effectively requiring no space.

[**extent1.cxx**](extent1.cxx) - [Compiler Explorer](https://godbolt.org/z/rz8jKKeGe)
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

mdspan is challenging to implement because the division of extents into static and dynamic categories creates an irregularity when indexing. This is most acute in the constructor which takes a function parameter for each dynamic extent. The Circle implementation defines a variable template `find_dynamic_index` which counts the number of dynamic extents prior to some extent index `I`. This is a search operation, easily accomplished by using a [pack slice](https://github.com/seanbaxter/circle/tree/master/universal#pack-subscripts-and-slices) inside a _fold-expression_. 

```cpp
  template<size_t I>
  static constexpr size_t find_dynamic_index =
    (0 + ... + (dynamic_extent == Extents...[:I]));
```

The slice `...[:I]` effectively truncates the template parameter pack `Extents` to its first `I` elements. These are compared with `dynamic_extent`, and all the matches are summed.

```cpp
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
```

The constructor's _mem-initializer-list_ initializes each element of the member pack `m`. If the extent corresponding to a member is `dynamic_extent`, then `find_dynamic_index<int...>`, where `int...` yields the index of the current pack expansion, specifies a gather index into the function parameter pack `exts`. We're telling the compiler: if this is a dynamic extent, search for the function parameter specifying the extent, and gather and initialize from that; otherwise, initialize from the static extent in the `Extents` template parameter.

The [constexpr conditional operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-conditional--) `??:` serves as an important guard. Consider if we specialized `extents<dynamic_extent, 3>` and called the one-parameter dynamic extent constructor. When computing the subobject initializer for `m1`, we might specialized `find_dynamic_index<1>`, which would count the number of preceding `dynamic_extent` elements, which is 1 in this case. `exts...[find_dyanmic_index<int...>]` would then attempt to read function parameter 1, which is out-of-range, because we only passed it one function parameter in all. However, the `??:` operator only substitutes the middle operand when the left-hand operand true, and subtitutes the right-hand operand when the left-hand operand is false

## 2. Circle Imperative Arguments.

Circle includes a simple [domain-specific language](https://github.com/seanbaxter/circle/tree/master/imperative#circle-imperative-arguments-cia) for programmatically constructing template argument lists, function argument lists and initializer lists with common programming primitives like for loops, if statements, and declarations. I don't know if it's Turing Complete, but you can write a [Game of Life](https://github.com/seanbaxter/circle/tree/master/imperative#conways-game-of-life) in a handful of lines, entirely within a template argument list.

### Create an n-length set.

mdspan specifies a `dextents` alias template which typedefs a fully-dynamic `extents` object for a given rank:

```cpp
template<size_t Rank>
using dextents = extents<dynamic_extent x Rank times here>;
```

The proposal gives this exposition implementation:

```cpp
  template<size_t Rank>
    using dextents = decltype(
      [] <size_t... Pack> (index_sequence<Pack...>) constexpr {
        return extents<
          [] (auto) constexpr { return dynamic_extent; } (
            integral_constant<size_t, Pack>{})...>{};
      }(make_index_sequence<Rank>{}));
```

The reference implementation uses a more [traditional TMP approach](https://github.com/kokkos/mdspan/blob/a32d60ac5632e340c6b991f37910fd7598ea07cf/mdspan.hpp#L3320), with recursive template specialization.

But this should be easy. You shouldn't need _nested lambdas_ in unevaluated contexts just to replicate an argument N times. This is where CIA makes programming easy:

```cpp
template<size_t Rank>
using dextents = extents<for i : Rank => dynamic_extent>;
```

Inside the _template-argument-list_ production, write an [_argument-for_](https://github.com/seanbaxter/circle/tree/master/imperative#argument-for) to loop the declaration `i` from 0 to Rank - 1. The `=>` wide arrow indicates the body of the loop, which can be a type, non-type, template, universal template parameter, or another CIA construct like an _if_, _for_, declaration, or so on. We can literally write loops in argument lists, and deposit arguments with each step.

### Tuple cat.

```cpp
template< class... Tuples >
std::tuple<CTypes...> tuple_cat(Tuples&&... args);
```

[std::tuple_cat](https://en.cppreference.com/w/cpp/utility/tuple/tuple_cat) is a generic function that concatenates all tuple elements in all of its arguments. It's [hard to implement](https://github.com/gcc-mirror/gcc/blob/7adcbafe45f8001b698967defe682687b52c0007/libstdc%2B%2B-v3/include/std/tuple#L1693)!

[**tuple_cat1.cxx**](tuple_cat1.cxx) - [Compiler Explorer](https://godbolt.org/z/W3nqYd5jb)
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

int main() {
  using namespace std::string_literals;
  auto t1 = std::make_tuple(1, 2.2, "Three");
  auto t2 = std::make_tuple("Four"s, 5i16);
  auto t3 = std::make_tuple(6.6f, 7ull);

  auto cat1 = tuple_cat1(t1, t2, t3);

  std::cout<< decltype(cat1).tuple_elements.string<< ": "<< cat1.[:]<< "\n" ...;  
}
```
```
$ circle tuple_cat1.cxx && ./tuple_cat1
int: 1
double: 2.2
const char*: Three
std::basic_string<char, std::char_traits<char>, std::allocator<char>>: Four
short: 5
float: 6.6
unsigned long long: 7
```

But CIA makes it easy. There are two steps:
1. Form the function return type by looping over all types in the `Tuples` template parameter pack, and expanding the trait `tuple_elements` on each one.
2. Form the _initializer-list_ for the return object by looping over all types in the template parameter pack, declaring `N` to hold the `tuple_size` of each parameter tuple, then calling `get` to extract each member from each function parameter. `get` is specialized over the pack index `int...(N)`, and expanded into the _initializer-list_.

* `for` _[step-decl ,]_  `typename`_decl_ `:` _type-pack_ `=>` _generic-argument_

We use [this syntax](https://github.com/seanbaxter/circle/tree/master/imperative#argument-for) for _argument-for_, which both declares an integer step index and a type alias for each element in the type parameter pack. Pack subscript `tpls...[i]` accesses each function parameter, and the `get` pack expansion disaggregates it.

Circle supports `tuple_elements` (and `variant_alternatives`) as pack-yielding member traits. Under the hood the compiler queries `tuple_size` and probes the `tuple_elements` class template for each element index, and exposes all this information to the sure as an imperative pack. There's nothing to deduce, you just ask and expand it right into the argument list.

## 3. Deduced forwarding references.

There's an in depth discussion of deduced forwarding references [here](../tuple#deduced-forwarding-references). In a word, deduced forwarding references fix a hole in C++ function declarations and overload resolution. They allow a function parameter to infer the reference and cv qualifiers of its argument and to actively deduce the argument type. This differs from ordinary C++11 forwarding references which infer all three things.

As an example:

```cpp
template<typename T, typename... Args>
void f(T&& u : std::tuple<Args...>);
```

`T` can deduced to any of eight possible types, allowing this one function to take the place of up to eight separate overloads:

```cpp
template<typename... Args>
void f(std::tuple<Args...>& u);

template<typename... Args>
void f(const std::tuple<Args...>& u);

template<typename... Args>
void f(volatile std::tuple<Args...>& u);

template<typename... Args>
void f(const volatile std::tuple<Args...>& u);

template<typename... Args>
void f(std::tuple<Args...>&& u);

template<typename... Args>
void f(const std::tuple<Args...>&& u);

template<typename... Args>
void f(volatile std::tuple<Args...>&& u);

template<typename... Args>
void f(const volatile std::tuple<Args...>&& u);
```

This is directly applicable to generic containers like `std::tuple`, which associated functions with four overloads differing only in their cv-ref qualifiers.

```cpp
template<size_t I, class Tuple, class...Types>
auto&& get(Tuple&& t : tuple<Types...>) noexcept {
  static_assert(I < sizeof...(Types));
  return std::forward<Tuple>(t).template _get<I>();
}

template<class T, class Tuple, class... Types>
auto&& get(Tuple&& t : tuple<Types...>) noexcept {
  static_assert(1 == (0 + ... + (T == Types)));
  constexpr size_t I = T == Types ...?? int... : -1;
  return std::forward<Tuple>(t).template _get<I>();
}
```

The Circle tuple [implementation](../tuple/tuple.hxx) reduces the eight `get` overloads declared in the Standard to just two function definitions. This 4:1 replacement can be observed in many scenarios in the Standard Library as well as user code. It plays nicely with [P0847R7 - Deducing 'this'](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0847r7.html), also implemented in Circle. That proposal exposes the implicit object argument as an explicit function parameter. You could already use a C++11 forwarding reference with that. But now you can use a deduced forwarding reference to further constrain it to the class type, getting around the ["shadowing problem"](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2481r0.html#the-shadowing-mitigation-private-inheritance-problem).

## 4. N-dimensional visit.

By far the most troublesome function in the C++ variant is the [std::visit](http://eel.is/c++draft/variant.visit) function. Experience with variant inspired a Circle builtin for generating an n-dimensional visitor pattern. Michael Park [documented](https://mpark.github.io/programming/2019/01/22/variant-visitation-v2/) the struggles of implementing even a one-dimensional visitor with ISO C++.

[**visit.hxx**](../variant/variant.hxx)
```cpp
template <class Visitor, class... Variants>
constexpr decltype(auto) visit(Visitor&& vis, Variants&&... vars) {
  if((... || vars.valueless_by_exception()))
    throw bad_variant_access("variant visit has valueless index");

  return __visit<Variants.remove_reference.variant_size...>(
    std::invoke(
      std::forward<Visitor>(vis), 
      std::forward<Variants>(vars).template get<indices>()...
    ),
    vars.index()...
  );  
}
```

The [`__visit`](../variant#visit) and `__visit_r` builtins take N template arguments, which indicate the _extent_ of each dimension. The first function parameter is the callable. The subsequence N function parameters indicate the runtime values for each dimension. The compiler generates the N-dimensional switch.

The `__visit` builtin is also reflection-aware, allowing specialization over enums (where it visits all enumerators), and on specializations of `integer_sequence`, where it visits all template parameters.

## 5. Builtins for overload resolution.

Variant inspired two extensions for accessing the compiler's overload resolution capabilities. 

`__preferred_copy_init` takes an argument type and a set of target types, and finds the target type with the best viable copy initialization given the argument type. If there is no viable initialization, or there are ambiguous best viable conversions, the builtin yields -1.

This makes for an easy implementation of the [variant converting constructor](http://eel.is/c++draft/variant#ctor-14).
[**variant.hxx**](../variant/variant.hxx)
```cpp
  template<typename T, int j = __preferred_copy_init(T, Types...)>
  requires(
    -1 != j &&
    T.remove_cvref != variant && 
    T.template != std::in_place_type_t &&
    T.template != std::in_place_index_t && 
    std::is_constructible_v<Types...[j], T>
  )
  constexpr variant(T&& arg) 
    noexcept(std::is_nothrow_constructible_v<Types...[j], T>) :
    m...[j](std::forward<T>(arg)), _index(j) { }
```

`__preferred_assignment` has the same parameterization as `__preferred_copy_init`, but finds the best viable assignment operator. This makes implementing the [variant converting assignment](http://eel.is/c++draft/variant#assign-11) very easy.

```cpp
  template<class T, size_t j = __preferred_assignment(T&&, Types...)>
  requires(T.remove_cvref != variant && -1 != j &&
    std::is_constructible_v<Types...[j], T>)
  constexpr variant& operator=(T&& t) 
  noexcept(std::is_nothrow_assignable_v<Types...[j], T> &&
    std::is_nothrow_constructible_v<Types...[j], T>) {
 
    if(_index == j) {
      // If *this holds Tj, assigns std::forward<T>(t) to the value contained
      // in *this.
      m...[j] = std::forward<T>(t);
 
    } else if constexpr(std::is_nothrow_constructible_v<Types...[j], T> ||
      !std::is_nothrow_move_constructible_v<Types...[j]>) {
 
      // Otherwise, if is_nothrow_constructible_v<Tj, T> || 
      // !is_nothrow_move_constructible_v<Tj> is true, equivalent to
      // emplace<j>(Tj(std::forward<T>(t))).
      reset();
      new(&m...[j]) Types...[j](std::forward<T>(t));
      _index = j;
 
    } else {
      // Otherwise, equivalent to emplace<j>(Tj(std::forward<T>(t))).
      Types...[j] temp(std::forward<T>(t));
      reset();
      new(&m...[j]) Types...[j](std::move(temp));
      _index = j;
    }
 
    return *this;
  }
```

## 6. First-class tuple support.

As documented [here](../universal#tuple-subscripts-and-slices), the operators `.[I]` and `[begin:end:step]` subscript and slice "structured types." What types are these?
* Tuple-like types that specialize `std::tuple_size` yield their elements.
    * `std::tuple`
    * `std::pair`
    * `std::array`
    * `circle::tuple`
* Builtin arrays yield their elements.
* Other classes and structs yield their non-static public data members.

The `sizeof.` operator returns the number of elements in a structured type. For a tuple-like type, this returns `std::tuple_size`. For an array, it returns the number of elements in the first rank. For other class types, it returns the number of non-static data members.

The `tuple_elements` member trait probes `std::tuple_elements` and yields the contained types as a type parameter pack. This is fully imperative, so you don't have to call a function that exposes template parameters that can be deduced to use it.

[**access.cxx**](access.cxx) - [Compiler Explorer](https://godbolt.org/z/PG9zqqeMx)
```cpp
#include <tuple>
#include <array>
#include <iostream>

int main() {
  std::tuple<int, double, const char*> tup(
    100, 3.14, "Hello std::tuple"
  );

  std::cout<< int...<<": "<< decltype(tup).tuple_elements.string<< "\n" ...;

  // Print out by subscript.
  std::cout<< "Print by subscript:\n";
  std::cout<< "  0: "<< tup.[0]<< "\n";
  std::cout<< "  1: "<< tup.[1]<< "\n";
  std::cout<< "  2: "<< tup.[2]<< "\n";

  // Print out by slice.
  std::cout<< "Print by slice - "<< sizeof. tup<< " elements:\n";
  std::cout<< "  "<< int...<< ": "<< tup.[:]<< "\n" ...;

  std::pair<const char*, long> pair(
    "A pair's string",
    42
  );
  std::cout<< "Works with pairs - "<< sizeof. pair<< " elements:\n";
  std::cout<< "  "<< int...<< ": "<< pair.[:]<< "\n" ...;

  int primes[] { 2, 3, 5, 7, 11 };
  std::cout<< "Works with builtin arrays - "<< sizeof. primes<< " elements:\n";
  std::cout<< "  "<< int...<< ": "<< primes.[:]<< "\n" ...;
}
```
```
$ circle access.cxx && ./access
0: int
1: double
2: const char*
Print by subscript:
  0: 100
  1: 3.14
  2: Hello std::tuple
Print by slice - 3 elements:
  0: 100
  1: 3.14
  2: Hello std::tuple
Works with pairs - 2 elements:
  0: A pair's string
  1: 42
Works with builtin arrays - 5 elements:
  0: 2
  1: 3
  2: 5
  3: 7
  4: 11
```

This sample shows the tuple subscript and slice operators on a variety of types. Because Circle imperatively will create a pack from a tuple's elements, you don't need indirection through an `apply`-like function for the purpose of argument deduction. Write your operation inline, and use a pack expansion at the end of the statement to transform each element.
