# Circle templates and packs

## Contents

1. [Template parameter kinds](#template-parameter-kinds)
1. [Class template specialization traits](#class-template-specialization-traits)
1. [Generic comparisons](#generic-comparisons)
1. [Universal parameter traits](#universal-parameter-traits)
1. [First class tuple and variant](#first-class-tuple-and-variant)

## Template parameter kinds

Circle supports six kinds of template parameters:

1. `typename Type` - type parameter.
2. _type_ `NonType`- non-type parameter.
3. `template<`_template-args_`> class Temp` - type template parameter.
4. `template<`_template-args_`> auto Var` - variable template parameter
5. `template<`_template-args_`> concept Concept` - concept parameter.
6. `template auto Universal` - universal parameter [P1985](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1985r1.pdf).

All six kinds support parameter packs by writing the pack token `...` before the parameter name.

[**params1.cxx**](params1.cxx)
```cpp
#include <iostream>
#include <type_traits>
#include <vector>
#include <concepts>

template<
  typename Type,
  auto NonType,
  template<template auto...> class Temp,
  template<template auto...> auto Var,
  template<template auto...> concept Concept,
  template auto Universal
>
void func() {
  // .string is stringification with compile-time reflection.
  std::cout<< "Type      = " + Type.string      + "\n";
  std::cout<< "NonType   = " + NonType.string   + "\n";
  std::cout<< "Temp      = " + Temp.string      + "\n";
  std::cout<< "Var       = " + Var.string       + "\n";
  std::cout<< "Concept   = " + Concept.string   + "\n";
  std::cout<< "Universal = " + Universal.string + "\n";
}

enum Shapes {
  square, triangle, circle
};

int main() {
  func<
    const char[16],       // Pass a type
    circle,               // Pass a non-type
    std::vector,          // Pass a type template
    std::is_enum_v,       // Pass a variable template
    std::integral,        // Pass a concept
    Shapes                // Pass anything to a universal parameter
  >();
}
```
```
$ circle params1.cxx -std=c++20
$ ./params1
Type      = const char[16]
NonType   = circle
Temp      = std::vector
Var       = std::is_enum_v
Concept   = std::integral
Universal = Shapes
```

Support for variable template and concept parameters make the language more consistent over supported parameterizations. Universal parameters help make everything more generic. Even if the top-level template parameter isn't a universal parameter, this feature may still be used when declaring the parameterization of the three template parameter kinds (type templates, variable templates and concepts).

[**tuple1.cxx**](tuple1.cxx)
```cpp
#include <tuple>
#include <concepts>

// Evaluate the concept C on each tuple_element of T.
// They must all evaluate true.
template<typename T, template<typename> concept C>
concept tuple_like_of = (... && C<T.tuple_elements>);

// Constrain func to tuple-like integral types.
void func(tuple_like_of<std::integral> auto tup) { }

int main() {
  func(std::tuple<int, unsigned short>()); // OK
  func(std::array<char, 5>());             // OK
  func(std::pair<short, char>());          // OK
  func(std::tuple<int, const char*>());    // Error
}
```
```
$ circle tuple1.cxx -std=c++20
error: void func(<#0>)
failure on overload resolution for function void func(<#0>)
  function declared at tuple1.cxx:10:6
tuple1.cxx:16:7
  func(std::tuple<int, const char*>());    // Error 
      ^
  instantiation: tuple1.cxx:10:6
  during constraints checking of function template void func(<#0>)
  template arguments: [
    '<#0>' = std::tuple<int, const char*>
      class 'tuple' declared at /usr/include/c++/10/tuple:891:5
  ]
  void func(tuple_like_of<std::integral> auto tup) { } 
       ^
    constraint: tuple1.cxx:10:24
    constraint tuple_like_of<std::integral> not satisfied over std::tuple<int, const char*>
      concept definition at tuple1.cxx:7:9
      constraint attached to param '<#0>' of template void func(<#0>)
      template declaration at tuple1.cxx:10:6
    void func(tuple_like_of<std::integral> auto tup) { } 
                           ^
```

Parameterizing concepts allows parametric type constraints. The concept `tuple_like_of` checks that all element types of a tuple-like object (i.e., a type that provides an `std::tuple_size` partial or explicit specialization) satisfies the parameter concept `C`. Use the `tuple_like_of` as a type constraint on function or template parameters, and specialize it with the concept of your choice--either something out of the Standard Library (here we use `std::integral`) or something custom.

## Class template specialization traits

Types support eight member traits for accessing information about class template specialization:

1. `.is_specialization` - is a class template specialization.
2. `.template` - the class template of a specialization. 
3. `.type_args` - the template arguments as type pack.
4. `.nontype_args` - the template arguments as a non-type pack.
5. `.template_args` - the template arguments as a type template pack.
6. `.var_template_args` - the template arguments as a variable template pack.
7. `.concept_args` - the template arguments as a concept pack.
8. `.universal_args` - the template arguments as a universal pack.


// NOTE: Merge and sort type lists.
// 

## Universal parameter traits

Universal parameters support five non-type member traits that provide useful introspection for filtering parameters before forwarding them to other templates:

1. `.is_type` - is a type argument.
2. `.is_nontype` - is a non-type argument.
3. `.is_template` - is a template template argument.
4. `.is_var_template` - is a variable template argument.
5. `.is_concept` - is a concept argument.

[**universal1.cxx**](universal1.cxx)
```cpp
#include <iostream>
#include <type_traits>
#include <concepts>
#include <vector>

// Takes a pack of type templates!
template<template<template auto...> class... Temps>
void f1() {
  std::cout<< "f1 type templates:\n";
  std::cout<< "  " + Temps.string + "\n" ...;
}

// Takes a pack of variable templates!
template<template<template auto...> auto... Vars>
void f2() {
  std::cout<< "f2 variable templates:\n";
  std::cout<< "  " + Vars.string + "\n" ...;
}

// Takes a pack of concepts!
template<template<template auto...> concept... Concepts>
void f3() {
  std::cout<< "f3 concepts:\n";
  std::cout<< "  " + Concepts.string + "\n" ...;
}

template<template auto... Params>
void g() {
  // Specialize f1 with only the type template parameters.
  f1<if Params.is_template => Params ...>();

  // Specialize f2 with only the variable template parameters.
  f2<if Params.is_var_template => Params ...>();

  // Specialize f3 with only the concept parameters.
  f3<if Params.is_concept => Params ...>();
}

int main() {
  // Specialize g with all sorts of things.
  g<
    int*,                // type
    300,                 // non-type
    std::vector,         // type template
    std::is_same_v,      // variable template
    std::is_same,        // type template
    std::integral,       // concept
    std::is_enum_v,      // variable template
    std::floating_point, // concept
    void                 // type
  >();
}
```
```
$ circle universal1.cxx -std=c++20
$ ./universal1
f1 type templates:
  std::vector
  std::is_same
f2 variable templates:
  std::is_same_v
  std::is_enum_v
f3 concepts:
  std::integral
  std::floating_point
```

Universal parameter member traits to indicate the kind of substituted argument. Use in conjunction with [Circle Imperative Arguments](https://github.com/seanbaxter/circle/tree/master/imperative#circle-imperative-arguments-cia) to effect control flow within template and function argument lists.

## Generic comparisons

Circle defines a generic comparison operators `==` or `!=` which work on all these kinds of operands:
* types
* non-types (that's a normal comparison)
* templates
* variable templates
* concepts
* universal parameters








## First class tuple and variant

C++ defines extension points for tuples and variants with the [`std::tuple_size`](https://en.cppreference.com/w/cpp/utility/tuple/tuple_size) and [`std::variant_size`](https://en.cppreference.com/w/cpp/utility/variant/variant_size) classes. Partial specializations for the Standard Library tuple and variant types ship with C++, and user-defined types can indicate they support a tuple- or variant-like interface by providing their own specializations.
