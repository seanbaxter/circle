#include <iostream>
#include <string>

int main() {
  std::string s = "hello";
  std::cout<< s[:]<< ' ' ...;   // Prints 'h e l l o '
  std::cout<< '\n';
}

