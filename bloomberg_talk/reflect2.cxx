#include <cstdio>
#include <string>
#include <iostream>

template<typename type_t>
void print_object(const type_t& obj) {
  std::cout<< 
    @type_string(@member_types(type_t))<< " "<< 
    @member_names(type_t)<< ": "<< 
    @member_values(obj)<< "\n" ...;
}

struct foo_t {
  int x;
  double y;
  std::string z;
};

int main() {
  @meta foo_t obj { 5, 3.14, "crazy string" };
  @meta print_object(obj);

  return 0;
}