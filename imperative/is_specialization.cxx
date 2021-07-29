#include <tuple>

using T1 = std::tuple<int, char>;
using T2 = std::pair<int, char>;

// P1985R1 Universal Template Parameters
template<typename T, template<template auto...> class Temp>
constexpr bool is_specialization_v = false;

template<template auto... Params, template<template auto...> class Temp>
constexpr bool is_specialization_v<Temp<Params...>, Temp> = true;

static_assert(is_specialization_v<T1, std::tuple>);
static_assert(!is_specialization_v<T2, std::tuple>);
