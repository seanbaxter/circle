#include <cstdio>

template<typename... types_t>
struct reverse_t {
  types_t...[::-1] @(int...) ...;
};

@meta puts(@member_decl_strings(reverse_t<int, float, char[4]>))...;