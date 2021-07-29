#include <cstdio>
#include <memory>
#include <utility>

template<typename...> struct list;

// Fill a list with a bunch of types.
using L = list<int, double, char*, double, double, float[5], float, 
  void(*)(void), void*, float, void*, wchar_t, int, float[5], 
  std::unique_ptr<int>, char(*)(float), std::pair<int, double>, char(*)(float),
  short, char8_t, float[5], int, int*, float[5]>;

// Print original types:
@meta printf("Original types:\n");
@meta printf("%2d: %s\n", int..., L.type_args.string)...;

// Unique and print them.
@meta printf("\nUnique types:\n");
@meta printf("%2d: %s\n", int..., L.type_args.unique.string)...;