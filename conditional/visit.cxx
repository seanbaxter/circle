#include <variant>
#include <tuple>
#include <array>
#include <iostream>

template<typename type_t>
auto call_first(const type_t& x, auto&&... fs) {
  return requires { fs(x); } ...?? fs(x) : static_assert(@type_string(type_t));
}

template<typename... types_t, typename... funcs_t>
auto visit1(const std::variant<types_t...>& variant, funcs_t&&... fs) {
  // Use a multi conditional operator and forward to call_first.
  return int...(sizeof...(types_t)) == variant.index() ...?
    call_first(std::get<int...>(variant), std::forward<funcs_t>(fs)...) :
    __builtin_unreachable();
}

template<typename... types_t>
auto visit2(const std::variant<types_t...>& variant, auto&&... fs) {
  // Generate a switch and use a ...?? in each case.
  switch(variant.index()) {
    @meta for(int i : sizeof...(types_t)) {
      case i:
        return requires { fs(std::get<i>(variant)); } ...?? 
          fs(std::get<i>(variant)) :
          static_assert(@type_string(types_t...[i]));
    }
  }
}


void f1(double x) { std::cout<< "f1: "<< x<< "\n"; }

void f2(const char* x) { std::cout<< "f2: "<< x<< "\n"; }

auto f3 = []<typename type_t, size_t I>(std::array<type_t, I> a) {
  std::cout<< "f3: ";
  std::cout<< a...[:]<< " " ...;
  std::cout<< "\n";
};

int main() {
  std::variant<
    double,
    const char*,
    std::array<int, 3>,
    std::array<double, 2>
  > v;
  
  v = 3.14;
  visit1(v, f1, f2, f3);

  v = "Hello";
  visit1(v, f1, f2, f3);
  
  v = std::array { 1, 2, 3 };
  visit2(v, f1, f2, f3);
}
