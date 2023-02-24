#feature on as
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

  // Opt into safety.
  #feature on no_implicit_pointer_to_bool
  
  // Error: no implicit conversion from const char* to bool.
  func("Hello world!");

  // Explicitly cast to a string. This works.
  func("Hello world!" as std::string);

  // We can opt-back into implicit conversions and match the bool overload.
  func("Hello world!" as _);
}