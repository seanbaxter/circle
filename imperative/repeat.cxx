template<template auto... Ts> struct list;

// Duplicate a universal parameter N times.
template<int N, template auto T>
using dup = list<.{T}(N)>;

using L1 = dup<5, int>;
static_assert(L1 == list<int, int, int, int, int>);

// Interleave the parameter with the type void.
template<int N, template auto T>
using alternate_void = list<T, .{void, T}(N - 1)>;

using L2 = alternate_void<3, char>;
static_assert(L2 == list<char, void, char, void, char>);

// Repeat a pack of arguments N times. Vary the Ts first. Note ... placement.
template<int N, template auto... Ts>
using repeat_inner = list<.{Ts...}(N)>;

using L3 = repeat_inner<2, char, float, long>;
static_assert(L3 == list<char, float, long, char, float, long>);

// Repeat of pack of arguments N times. Vary the Ns first. Note ... placement.
template<int N, template auto... Ts>
using repeat_outer = list<.{Ts}(N)...>;

using L4 = repeat_outer<2, char, float, long>;
static_assert(L4 == list<char, char, float, float, long, long>);
