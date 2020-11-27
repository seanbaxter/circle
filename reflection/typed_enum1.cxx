#include <cstdio>

enum typename my_types_t {
  double,
  int,
  char*,
  int[5],
  char,
};

template<typename type_t>
void print_enum_types1() {
  printf("%s (for loop):\n", @type_string(type_t));

  // Use a for loop.
  @meta for(int i = 0; i < @enum_count(type_t); ++i)
    printf("  %s\n", @enum_type_string(type_t, i));
}

template<typename type_t>
void print_enum_types2() {
  printf("%s (for-enum):\n", @type_string(type_t));

  // Use a for-enum loop.
  @meta for enum(type_t e : type_t)
    printf("  %s\n", @enum_type_string(e));
}

template<typename type_t>
void print_enum_types3() {
  printf("%s (pack):\n", @type_string(type_t));
  printf("  %s\n", @enum_type_strings(type_t))...;
}

int main() {
  print_enum_types1<my_types_t>();
  print_enum_types2<my_types_t>();
  print_enum_types3<my_types_t>();
}