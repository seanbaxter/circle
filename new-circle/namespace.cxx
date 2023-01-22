#include <iostream>

// Version your code by namespace.
namespace v1 { 
  void func() {
    std::cout<< "Called v1::func()\n";
  }
}

namespace v2 { 
  void func() {
    std::cout<< "Called v2::func()\n";
  }
}

// Parameterize a function template over a namespace.
template<namespace ns>
void f() {
  // Qualified dependent name lookup.
  ns::func();
}

int main() {
  // Specialize the template based on version.
  f<v1>();
  f<v2>();
}
