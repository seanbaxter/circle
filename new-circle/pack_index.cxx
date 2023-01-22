#include <iostream>

template<int I>
void func() {
  std::cout<< "Got index "<< I<< "\n";
}

int main() {
  // Call func once for each index {0, 1, 2, 3, 4}
  func<int...(5)>() ...;

  // Or just do it directly, with no function call.
  std::cout<< "Even better "<< int...(5)<< "\n" ...;
}