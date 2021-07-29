#include <map>

template<typename T>
using index_and_arg = T.template<
  T.template<int..., T.universal_args>...
>;

template<template auto...> struct list;

using L = list<int, 100, std::map>;

static_assert(index_and_arg<L> == list<
  list<0, int>, list<1, 100>, list<2, std::map>
>);