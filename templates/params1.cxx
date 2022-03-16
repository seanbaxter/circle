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