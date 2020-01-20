#include <cstdio>

template<typename type_t>
void print_structure() {
  printf("%s:\n", @type_string(type_t));
  printf("  %s\n", @member_decl_strings(type_t))...;
}

template<typename... types_t>
struct tuple_t {
  types_t @(int...)...;
};

template<typename... types_t>
struct reverse_tuple_t {
  types_t...[::-1] @(int...)...;
};

@meta print_structure<tuple_t<int, char, double*> >();
@meta print_structure<reverse_tuple_t<int, char, double*> >();

int main() { }

