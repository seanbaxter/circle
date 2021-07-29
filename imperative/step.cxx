#include <map>

template<typename T>
using index_and_arg = T.template<
  for i, universal U : T.universal_args =>
    T.template<i, U>
>;

template<template auto...> struct list;

using L = list<int, 100, std::map>;

static_assert(index_and_arg<L> == list<
  list<0, int>, list<1, 100>, list<2, std::map>
>);
