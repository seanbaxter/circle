#include <iostream>

struct pair_t {
  int first;
  double second;
};

void func(auto&& obj) {
  std::cout<< decltype(obj)~string + "\n";
  std::cout<< decltype(obj.first)~string + "\n";
}

int main() {
  pair_t pair { 1, 2 };
  func(pair);
}