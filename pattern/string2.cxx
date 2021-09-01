#include <iostream>

template<typename type_t>
void f(const type_t& x) {
  std::cout<< type_t.string + ": "<< std::flush;
  
  inspect(x) {
    // Anything convertible to string, except nullptr!
    s is not nullptr as std::string     => std::cout<< "string "<< s<< "\n";

    is _                                => std::cout<< "unsupported type\n";
  }
}

int main() {
  f("Hello world");
  f(nullptr);
}