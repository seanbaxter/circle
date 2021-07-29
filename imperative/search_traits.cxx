#include <type_traits>

template<template auto... Xs> struct list;

// Search for the left-most integral argument.
template<typename T>
constexpr int FirstIntegral = T.type_args.is_integral ...?? 
  int... : sizeof...(T.type_args);

using L1 = list<double, void, short, float>;
static_assert(2 == FirstIntegral<L1>);

// Search for the right-most rvalue reference argument.
// Use the static slice ...[::-1]to reverse the order of T.type_args.
template<typename T>
constexpr int LastRValue = sizeof...(T.type_args) - 
  (T.type_args...[::-1].is_rvalue_reference ...?? 1 + int... : 0);

using L2 = list<void, char&&, float&, int&&, double*, int>;
static_assert(3 == LastRValue<L2>);