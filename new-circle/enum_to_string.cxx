#include <iostream>

template<typename T> requires(T~is_enum)
const char* enum_to_string(T e) {
  //   return
  //     circle   == e ? "circle"           :
  //     line     == e ? "line"             :
  //     triangle == e ? "triangle"         :
  //     square   == e ? "square"           :
  //     pentagon == e ? "pentagon"         :
  //                     "unknown<shapes_t>";
  return T~enum_values == e ...?
    T~enum_names : 
    "unknown <{}>".format(T~string);
}

enum shapes_t {
  circle, line, triangle, square, pentagon,
};

int main() {
  shapes_t shapes[] { line, square, circle, (shapes_t)10 };
  std::cout<< enum_to_string(shapes[:])<< "\n" ...;
}