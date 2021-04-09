#include <iostream>
#include <functional>
#include <tuple>
#include <array>

void func(auto... args) {
  std::cout<< args<< " "...;
  std::cout<< "\n";
}

template<auto... args>
struct foo_t { 
  foo_t() {
    std::cout<< @type_string(foo_t)<< "\n";
  }
};

int main() {
  // Expand array into a function argument list.
  constexpr int data1[] { 1, 2, 3 };
  func(0, data1..., 4);

  // Expand a normal array into an std::array.
  // Expand std::array into a function argument list.
  constexpr std::array data2 { data1..., 4, 5, 6 };
  func(data2..., 7);

  // Expand a tuple into a funtion argument list.
  auto tuple = std::make_tuple('a', 2u, 300ll);
  func(tuple...);

  // Use in a unary fold expression.
  int max = (... std::max data1);
  std::cout<< "max = "<< max<< "\n";

  int product = (... * data2);
  std::cout<< "product = "<< product<< "\n";

  // Specialize a template over compile-time data.
  // It can be constexpr.
  constexpr int data[] { 10, 20, 30 };
  foo_t<data...> obj1;

  // Or it can be meta.
  struct bar_t {
    int a;
    long b;
    char32_t c;
  };
  @meta bar_t bar { 100, 200, U'A' };

  // meta objects are mutable.
  @meta bar.b++;

  foo_t<bar...> obj2;
}