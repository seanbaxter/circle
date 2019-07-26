#include <iostream>

enum class my_enum_t {
  a, b, c, d, e
};

// Create a class template.
template<my_enum_t e> 
struct my_class_t {
  int func() { return 1 + (int)e; }
};

// Define an explicit specialization over b.
template<> 
struct my_class_t<my_enum_t::b> {
  int func() { return 50; }
};


template<typename type_t>
int foo() {
  int result = (1 + ... + my_class_t<@enum_pack(type_t)>().func());  
  return result;
}

int main() {
  // Instantiate my_class_t specialized over each enumerator, call
  // func(), and add the results up.

  // int result = (1 + ... + my_class_t<@enum_pack(my_enum_t)>().func());
  std::cout<< "We got "<< foo<my_enum_t>()<< "\n";

  return 0;
}
