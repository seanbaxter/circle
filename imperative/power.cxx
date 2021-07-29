#include <cstdio>

template<typename... Ts>
struct list;

template<template auto... Xs>
using power_set = list<
  auto N : sizeof...(Xs) => 
    for i : (1<< N) => list<
      if i & (1<< int...(N)) => Xs ...
    >
>;

using L = power_set<char, int, float, long>;

static_assert(L == list<
  list<>, list<char>, list<int>, list<char, int>,
  list<float>, list<char, float>, list<int, float>, list<char, int, float>,
  list<long>, list<char, long>, list<int, long>, list<char, int, long>,
  list<float, long>, list<char, float, long>, list<int, float, long>,
  list<char, int, float, long>
>);