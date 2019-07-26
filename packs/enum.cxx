#include <iostream>
#include <iomanip>

enum typename class my_list_t {
  a = int,
  b = double,
  c = const char*,
  d = float(*)(float, float)
};

template<typename type_t>
void print_typed_enum() {
  static_assert(__is_typed_enum(type_t),
    "argument to print_typed_enum must be a typed enum");

  std::cout<< "Definition of type_t from print_typed_enum:\n";

  std::cout<< 
    std::setw(30)<< std::left<< 
    @decl_string(@enum_types(type_t), @enum_names(type_t))<< " : "<<
    (int)@enum_pack(type_t)<< "\n" ...;
}

int main() {
  std::cout<< "Definition of my_list_t bar from main():\n";

  // Try this in a non-dependent context.
  std::cout<< 
    std::setw(30)<< std::left<< 
    @decl_string(@enum_types(my_list_t), @enum_names(my_list_t))<< " : "<<
    (int)@enum_pack(my_list_t)<< "\n" ...;
  std::cout<< "\n";

  // Try it as a template.
  print_typed_enum<my_list_t>();

  return 0;
}