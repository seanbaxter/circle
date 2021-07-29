#include <array>

// Make a list of universal template parameters.
template<template auto... Xs> struct list;

// Find the index of the first argument that matches the key.
template<template auto Key, typename L>
constexpr size_t FindFirst = Key == L.universal_args ...?? 
  int... : sizeof... L.universal_args;

using L1 = list<int, 1, char*, std::array, 'x'>;

// Match types with ==.
static_assert(FindFirst<char*, L1> == 2);

// Match templates with ==.
static_assert(FindFirst<std::array, L1> == 3);

// Match non-types with ==.
static_assert(FindFirst<'x', L1> == 4);