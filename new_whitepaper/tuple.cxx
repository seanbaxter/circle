#include <iostream>

template<typename... types_t>
struct tuple_t {
  // Loop over all members of the parameter pack.
  // Declare a non-static data member for each pack element.
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};

int main() {
  tuple_t<int, double, const char*> my_tuple {
    5, 1.618, "Hello tuple"
  };

  std::cout<< my_tuple._0<< "\n";
  std::cout<< my_tuple._1<< "\n";
  std::cout<< my_tuple._2<< "\n";

  return 0;
}