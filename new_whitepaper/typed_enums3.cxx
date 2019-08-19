#include <iostream>
#include <algorithm>
#include <vector>

template<typename type_t>
void stable_unique(std::vector<type_t>& vec) {
  std::vector<type_t> vec2;
  for(type_t x : vec) {
    if(vec2.end() == std::find(vec2.begin(), vec2.end(), x))
      vec2.push_back(x);
  }
  vec = std::move(vec2);
}

enum typename type_list1_t {
  int,
  double*,
  char*,
  char32_t,
};

enum typename type_list2_t {
  void**,
  int[3],
  double*,
  int[3],
  char32_t
};

enum typename uniqued_t {
  // Make a std::vector<@mtype> holding all associated types from 
  // type_list1_t and type_list2_t.
  // The types array has automatic storage duration, determined by the
  // braces of the enum-specifier.
  @meta std::vector<@mtype> types {
    @dynamic_type(@enum_types(type_list1_t))...,
    @dynamic_type(@enum_types(type_list2_t))...
  };

  // Use STL algorithms to unique this vector.
  @meta stable_unique(types);

  // Declare a typed enum with these associated types.
  @pack_type(types)...;
};

// Print all the associated types in uniqued_t
@meta std::cout<< @type_name(@enum_types(uniqued_t))<< "\n" ...;

int main() {
  return 0;
}
