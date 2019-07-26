#include <iostream>
#include <iomanip>
#include <type_traits>

template<typename type_t>
void print_object(const type_t& obj) {
  static_assert(std::is_class<type_t>::value, 
    "argument to print_object must be a class object");

  std::cout<< "Value of type_t obj from print_object():\n";

  // Print the member declarator and the member name using pack expansion.
  std::cout<< 
    std::setw(20)<< std::left<< 
    @decl_string(@member_types(type_t), @member_names(type_t))<< " : "<<
    @member_pack(obj)<< "\n" ...;
}

struct bar_t {
  int a;
  double b;
  std::string c;
};

int main() {
  bar_t bar { 5, 3.14, "Hello" };

  // Try this in a non-dependent context.
  std::cout<< "Value of bar_t bar from main():\n";
  std::cout<< 
    std::setw(20)<< std::left<< 
    @decl_string(@member_types(bar_t), @member_names(bar_t))<< " : "<<
    @member_pack(bar)<< "\n" ...;
  std::cout<< "\n";

  // Try it as a template.
  print_object(bar);

  return 0;
}