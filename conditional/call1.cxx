#include <iostream>

void call(auto f, const auto& x) {
  requires { f(x); } ?? 
    std::cout<< f(x)<< "\n" : 
    std::cout<< "Could not call f("<< @type_string(decltype(x))<< ")\n";
}

int f(int x) { return x * x; }

int main() {
  call(f, 5);
  call(f, "Hello");
}