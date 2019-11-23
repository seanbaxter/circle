#include <iostream>

// "enum typename" starts a typed enum. This includes an optional identifier
// and a mandatory type-id. Use @enum_type to extract the associated type
// of a typed enum.
enum typename type_list_t {
  int,
  double*, 
  char*, 
  void, 
  float[4]
};

int main() {
  // Walk through the enum and print the associated types.
  std::cout<< "type_list_t with a for-enum loop:\n";
  @meta for enum(auto e : type_list_t)
    std::cout<< @type_string(@enum_type(e))<< "\n"; 
  std::cout<< "\n";

  // We can do the same thing with an parameter pack.
  std::cout<< "type_list_t with a pack expansion:\n";
  std::cout<< @type_string(@enum_types(type_list_t))<< "\n" ...;
  
  return 0;
}