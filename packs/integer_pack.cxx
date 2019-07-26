#include <utility>
#include <cstdio>

template<typename type_t>
struct reverse_args_t;

template<template<typename...> class temp, typename... types_t>
struct reverse_args_t<temp<types_t...> > {
  enum { count = sizeof...(types_t) };
  typedef temp<types_t...[count - 1 - __integer_pack(count)] ...> type_t;
};

template<typename... types_t>
struct tuple_t {
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};

int main() {
  tuple_t<int, double*, char[3], float> a;
  typename reverse_args_t<decltype(a)>::type_t b;

  @meta printf("a is %s\n", @type_name(decltype(a), true));
  @meta printf("b is %s\n", @type_name(decltype(b), true));

  return 0;
}