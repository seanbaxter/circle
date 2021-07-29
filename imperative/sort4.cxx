#include <type_traits>
#include <cstdio>

template<template auto...> struct list;

template<typename T>
using sort_and_filter = list<
  T.type_args.filter(!_0.is_empty).sort(sizeof(_0) > sizeof(_1))...
>;

template<typename... Ts>
constexpr bool is_reverse_sorted = 
  (... && (sizeof(Ts...[:-2]) >= sizeof(Ts...[1:])));

template<int size>
struct dummy_t {
  char x[size];
};

struct empty_t { };

// Start with a mix of empties and unsorted types.
using Unsorted = list<
  dummy_t<59>, empty_t,     dummy_t<22>, dummy_t<61>, dummy_t<28>,
  empty_t,     dummy_t<18>, dummy_t<59>, dummy_t<25>, dummy_t<75>,
  dummy_t<22>, dummy_t<70>, empty_t,     dummy_t<30>, empty_t,
  dummy_t<12>, dummy_t<37>, dummy_t<27>, empty_t,     dummy_t<84>
>;

// Discard the empties and sort the remaining types by decreasing size.
using Sorted = list<
  dummy_t<84>, dummy_t<75>, dummy_t<70>, dummy_t<61>, dummy_t<59>, 
  dummy_t<59>, dummy_t<37>, dummy_t<30>, dummy_t<28>, dummy_t<27>, 
  dummy_t<25>, dummy_t<22>, dummy_t<22>, dummy_t<18>, dummy_t<12>
>;

static_assert(Sorted == sort_and_filter<Unsorted>);


