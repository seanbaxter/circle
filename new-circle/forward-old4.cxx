#include <iostream>

#define CPP2_FORWARD(x) std::forward<decltype(x)>(x)

struct pair_t {
  int first;
  double second;
};

void print(auto&&... args) {
  std::cout<< "  " + decltype(args)~string ...;
  std::cout<< "\n";
}

void func(auto&& obj) {
  print(CPP2_FORWARD(obj.first), CPP2_FORWARD(obj.second));   // BAD!
  print(CPP2_FORWARD(obj).first, CPP2_FORWARD(obj).second);   // GOOD!
}

int main() {
  std::cout<< "Pass by lvalue:\n";
  pair_t obj { 1, 2.2 };
  func(obj);

  std::cout<< "Pass by rvalue:\n";
  func(pair_t { 3, 4.4 });
}