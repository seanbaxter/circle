#include <iostream>

constexpr bool even(auto x) {
  return 0 == (x % 2);
}

template<typename type_t>
void f(const type_t& x) {
  std::cout<< type_t.string + ": ";
  
  inspect(x) {
    // omitting other inspect-clauses.
    i is even                     => std::cout<< "even "<< i<< "\n";
  }
}

int main() {
  f(10);
  f("Hello world");
}