#include <cstdio>
#include <string>
#include <iostream>
#include <vector>

template<typename type_t>
void print_type() {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< 
    @type_string(@member_types(type_t))<< " "<< 
    @member_names(type_t)<< "\n" ...;
}

template<typename type_t>
struct struct_of_pointers_t {
  @meta for(int i = 0; i < @member_count(type_t); ++i)
    @member_type(type_t, i)* @(@member_name(type_t, i));
};

template<typename type_t>
struct struct_of_vectors_t {
  @meta for(int i = 0; i < @member_count(type_t); ++i)
    std::vector<@member_type(type_t, i)> @(@member_name(type_t, i));
};

struct foo_t {
  int x;
  double y;
  std::string z;
};

int main() {
  typedef struct_of_pointers_t<foo_t> foo2_t;
  typedef struct_of_vectors_t<foo_t> foo3_t;

  @meta print_type<foo2_t>();
  @meta print_type<foo3_t>();

  return 0;
}