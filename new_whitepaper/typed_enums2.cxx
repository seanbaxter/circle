#include <iostream>

enum typename type_list1_t {
  // Circle enums may be specified with semicolon-separated statements.
  int;
  double*;
  char*;
};

enum typename type_list2_t {
  // or comma-separation declarations.
  void**,
  int[3],
  char32_t
};

// Make a typed enum that joins type_list1_t and type_list2_t.
enum typename joined_t {
  // We can programmatically inject with compile-time control flow
  @meta for enum(auto e : type_list1_t)
    @enum_type(e);

  // Or with parameter pack trickery.
  @enum_types(type_list2_t)...;
};

// Print all the associated types in joined_t
@meta std::cout<< @type_name(@enum_types(joined_t))<< "\n" ...;

int main() {
  return 0;
}
