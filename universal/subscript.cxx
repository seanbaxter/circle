#include <utility>
#include <tuple>
#include <array>
#include <iostream>

typedef float __attribute__((vector_size(16))) vec4;

template<typename type_t>
void print_object1(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  @meta for(int i = 0; i < sizeof...(type_t); ++i)
    std::cout<< "  "<< i<< ": "<< obj...[i]<< "\n";
}

template<typename type_t>
void print_object2(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int...<< ": "<< obj...[0:-1:1]<< "\n" ...;
}

template<typename type_t>
void print_object3(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";

  // Write comma-separated members inside braces.
  std::cout<< "  { "<< obj...[0];
  std::cout<< ", "<< obj...[1:]...;
  std::cout<< " }\n";
}

template<typename type_t>
void print_object4(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  if constexpr(__is_structured_type(type_t)) {
    std::cout<< "  { "<< obj...[0];
    std::cout<< ", "<< obj...[1:]...;
    std::cout<< " }\n";
  } else
    std::cout<< "  "<< obj<< "\n";
}

int main() {
  // std::pair is tuple-like.
  print_object1(std::make_pair(1, "Hello pair"));

  // std::tuple is tuple-like.
  print_object2(std::make_tuple(2, 1.618, "Hello tuple"));

  // std::array is tuple-like.
  print_object3(std::array { 5, 10, 15 } );

  // builtin arrays are structured binding types.
  int array[] { 20, 25, 30 };
  print_object4(array);

  // builtin vectors are structured binding types.
  print_object4(vec4(10, 20, 30, 40));

  // print a scalar. Rely on the __is_structure_type trait to choose the
  // right behavior.
  print_object4(100);
}