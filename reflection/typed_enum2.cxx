#include <cstdio>

// Define four typed enums. We'll programmatically join these.
enum typename list1_t {
  double,
  int[5],
  char,
};

enum typename list2_t {
  int[5],
  bool&&,
  std::nullptr_t,
};

enum typename list3_t {
  void*,
  char16_t,
  char(*)(double),
};

enum typename list4_t {
  char,
  short,
  int,
};

// Join the two.
enum typename joined_list_t {
  // Declare individual types.
  float, double, ptrdiff_t;

  // Use a for loop.
  @meta for(int i = 0; i < @enum_count(list1_t); ++i)
    @enum_type(list1_t, i);

  // Use a for-enum loop.
  @meta for enum(list2_t e : list2_t)
    @enum_type(e);

  // Use a pack expansion.
  @enum_types(list3_t)...;

  // Use a pack expansion and reverse the order of elements.
  @enum_types(list4_t)...[::-1] ...;
};

int main() {
  printf("%2d: %s\n", int..., @enum_type_strings(joined_list_t))...;
}