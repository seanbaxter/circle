#include <iostream>
#include <array>
#include <utility>

auto call_first(auto&& x, auto&&... fs) {
  return requires { fs(x); } ...?? 
    fs(x) :     
    static_assert(@type_string(decltype(x)));
}

void f1(double x) { std::cout<< "f1: "<< x<< "\n"; }

void f2(const char* x) { std::cout<< "f2: "<< x<< "\n"; }

auto f3 = []<typename type_t, size_t I>(std::array<type_t, I> a) {
  std::cout<< "f3: ";
  std::cout<< a...[:]<< " " ...;
  std::cout<< "\n";
};

int main() {
  call_first("Hello ??", f1, f2, f3);

  // This causes an error:
  // call_first(std::pair(1, 2), f1, f2, f3);
}