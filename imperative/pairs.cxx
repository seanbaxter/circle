#include <map>
#include <array>
#include <cstdio>

// Use nested argument-for to emit all pairs of template arguments.
template<typename T1, typename T2> 
using all_pairs = T1.template<
  for universal A1 : T1.universal_args =>
    for universal A2 : T2.universal_args =>
      T1.template<A1, A2>
>;

// Use an argument-for to iterate over the outer argument list.
// Use a pack expansion to generate arguments over the inner list.
template<typename T1, typename T2> 
using all_pairs2 = T1.template<
  for universal A1 : T1.universal_args =>
    T1.template<A1, T2.universal_args>...
>;

template<template auto...> struct list;

// Fill with a type, a non-type and a template argument.
using T1 = list<int, 5, std::map>;
using T2 = list<double, 'a', std::array>;

// The outer product of T1 and T2.
using R = list<
  list<int, double>, list<int, 'a'>, list<int, std::array>, 
  list<5, double>, list<5, 'a'>, list<5, std::array>,
  list<std::map, double>, list<std::map, 'a'>, list<std::map, std::array>
>;

static_assert(all_pairs<T1, T2> == R);
static_assert(all_pairs2<T1, T2> == R);
