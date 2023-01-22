#include <iostream>

template<typename... Ts>
void func(Ts... x) {
  // Print the first parameter type and value.
  std::cout<< Ts...[0]~string<< " "<< x...[0]<< "\n";
  
  // Print the last parameter type and value.
  std::cout<< Ts...[-1]~string<< " "<< x...[-1]<< "\n";
}

int main() {
  func(10i16, 20i32, 30i64, 40.f, 50.0);
}