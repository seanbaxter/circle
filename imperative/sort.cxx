#include <cstdio>

template<template auto...>
struct list;

// Two steps for sorting.
// 1. For each element i, count how many elements j precede element i.
// 2. Search for the element with insertion index i and emit it.
template<int... x>
using sort_ints = list<
  auto ...insertions : { 
    for i : sizeof... x => (... + (x < x...[i] || x == x...[i] && int... < i))
  } => for i : sizeof... x =>
    if insertions == i => x ...
>;

// Compare the slice starting at 0 with the slice starting at 1.
template<int... x>
constexpr bool is_sorted = (... && (x...[0:-2] <= x...[1:]));

using Unsorted = list<
  59, 22, 61, 28, 62, 74, 42, 98, 45, 75, 
  18, 59, 25, 75, 20, 72, 48, 46, 11, 11, 
  22, 70, 30,  9, 85, 39, 78, 24, 91, 95,
  12, 37, 27, 84, 77, 73, 51, 39, 54, 45,
  56, 88, 37, 24, 58, 90, 37, 36, 11, 86,
   6, 82, 47, 90, 89, 79, 38, 88,  2, 39,
  87, 80, 32, 55, 42, 31, 64, 29, 31, 55, 
  35, 11, 34, 28,  3, 74, 63, 11, 60,  6,
  85, 19, 48, 33, 52, 94, 17, 64, 72, 21,
  99, 58, 96, 54, 10, 24, 31, 78, 76, 37
>;

using Sorted = list<
   2,  3,  6,  6,  9, 10, 11, 11, 11, 11, 
  11, 12, 17, 18, 19, 20, 21, 22, 22, 24, 
  24, 24, 25, 27, 28, 28, 29, 30, 31, 31, 
  31, 32, 33, 34, 35, 36, 37, 37, 37, 37, 
  38, 39, 39, 39, 42, 42, 45, 45, 46, 47, 
  48, 48, 51, 52, 54, 54, 55, 55, 56, 58, 
  58, 59, 59, 60, 61, 62, 63, 64, 64, 70, 
  72, 72, 73, 74, 74, 75, 75, 76, 77, 78, 
  78, 79, 80, 82, 84, 85, 85, 86, 87, 88, 
  88, 89, 90, 90, 91, 94, 95, 96, 98, 99
>;

static_assert(is_sorted<Sorted.nontype_args...>);
static_assert(Sorted == sort_ints<Unsorted.nontype_args...>);