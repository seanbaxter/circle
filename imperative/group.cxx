#include <type_traits>
#include <cstdio>

template<template auto... Ts> struct list;

// Emit integral and pointer-to-integral types using if-expression.
template<typename... Ts>
using integral_pointers = list<if Ts.is_integral => .{ Ts, Ts* } ...>;

using L1 = integral_pointers<float, int, char, void>;
static_assert(L1 == list<int, int*, char, char*>);

// Use grouping to emit a triangle shape of arguments.
// At each step, emit a non-type integer argument, i + 1, and i + 1
// types.
template<int N>
using Arrays = list<
  for i : N => .{ i + 1, for j : i + 1 => char[i + 1] }
>;

using L2 = Arrays<4>;

static_assert(L2 == list<
  1, char[1], 
  2, char[2], char[2], 
  3, char[3], char[3], char[3],
  4, char[4], char[4], char[4], char[4]
>);