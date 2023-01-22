#include <functional>
#include <iostream>

#define CPP2_FORWARD(x) std::forward<decltype(x)>(x)

void func2(auto&& x) {
  std::cout<< decltype(x)~string + "\n";
}

void func1(auto&& x) {
  func2(CPP2_FORWARD(x));
}

int main() {
  int x = 1;
  func1(x);
  func2(1);
}