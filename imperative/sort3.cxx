#include <cstdio>
#include <memory>
#include <utility>

template<typename...> struct list;

// .sort is a trait metafunction. Feed it a predicate. _0 and _1 are 
// dependent type declarations of the left and right types in a comparison.
// The sort is implemented in the compiler frontend, so it's fast.

// Fill a list with a bunch of types.
using L = list<int, double, char*, float[5], float, void(*)(void), void*, 
  wchar_t, int, std::unique_ptr<int>, char(*)(float), std::pair<int, double>,
  short, char8_t, int*, double[5]>;

// This alias template sorts argument types lexicographically.
template<typename T>
using sort_lex = T.template<T.type_args.sort(_0.string < _1.string)...>;

// This alias template sorts by size, with biggest first.
template<typename T>
using sort_size = T.template<T.type_args.sort(sizeof(_0) > sizeof(_1))...>;

// Sort and print in one expression.
@meta printf("Types sorted by string:\n");
@meta printf("%2d: %s\n", int..., sort_lex<L>.type_args.string)...;

@meta printf("\nTypes sorted by decreasing size:\n");
@meta printf("%2d: %s\n", int..., sort_size<L>.type_args.string)...;