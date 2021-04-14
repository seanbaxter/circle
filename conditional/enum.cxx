#include <cstdio>
#include <iostream>

template<typename enum_t>
const char* enum_to_name1(enum_t e) {
  switch(e) {
    @meta for enum(enum_t e2 : enum_t) {
      case e2:
        return @enum_name(e2);
    }
    default:
      return "<unknown>";
  }
}

template<typename enum_t>
const char* enum_to_name2(enum_t e) {
  return @enum_values(enum_t) == e ...? @enum_names(enum_t) : "<unknown>";
}

enum shapes_t {
  circle, square, rectangle = 100, octagon
};

int main() {
  std::cout<< enum_to_name1(square)<< "\n";
  std::cout<< enum_to_name2(rectangle)<< "\n";
  std::cout<< enum_to_name2((shapes_t)102)<< "\n";
}