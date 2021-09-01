#include <iostream>

template<typename type_t>
void f(const type_t& x) {
  std::cout<< type_t.string + ": ";
  
  inspect(x) {
    // Any object that compares equal with nullptr. This will be nullptr_t
    // or any null-valued pointer type.
    // is nullptr                    => std::cout<< "nullptr\n";

    // Anything convertible to string.
    s as std::string              => std::cout<< "string "<< s<< "\n";

    is _                          => std::cout<< "unsupported type\n";
  }
}

int main() {
  f("Hello world");
  f(nullptr);
}