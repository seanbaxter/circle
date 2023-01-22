#include <iostream>
#include <string>

void func(const std::string& s) {
  std::cout<< s<< "\n";
}

void func(bool b) {
  std::cout<< (b ? "true" : "false") << "\n";
}

int main() {
  // Prints "true"!!! We wanted the std::string overload!
  func("Hello world!");
}