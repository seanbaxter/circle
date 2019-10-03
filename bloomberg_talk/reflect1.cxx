#include <cstdio>
#include <string>
#include <iostream>

template<typename type_t>
void print_object(const type_t& obj) {
  @meta for(int i = 0; i < @member_count(type_t); ++i) {
    std::cout<< 
      @type_name(@member_type(type_t, i))<< " "<< 
      @member_name(type_t, i)<< ": "<< 
      @member_ref(obj, i)<< "\n";
  }
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