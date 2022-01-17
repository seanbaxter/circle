#include <iostream>
#include <utility>

template<typename... Ts>
struct tuple {
  Ts ...m;
};

template<typename T1>
void f1(T1&& x) {
  std::cout<< "  f1: "<< T1.string<< "\n";
} 

template<typename T2, typename... Args>
void f2(T2&& y : tuple<Args...>) {
  std::cout<< "  f2: "<< T2.string<<" | Args: ";
  std::cout<< Args.string<< " "...;
  std::cout<< "\n";
}

struct derived_t : tuple<int, char, void*> { };

int main() {
  derived_t d1;
  const derived_t d2;

  std::cout<< "lvalue:\n";
  f1(d1);
  f2(d1);

  std::cout<< "const lvalue:\n";
  f1(d2);
  f2(d2);

  std::cout<< "xvalue:\n";
  f1(std::move(d1));
  f2(std::move(d1));

  std::cout<< "const xvalue:\n";
  f1(std::move(d2));
  f2(std::move(d2));
}