#include <iostream>
#include <utility>

struct B1 {
  template<typename Self>
  auto&& get0(this Self&& self) {
    // Error: ambiguous declarations found from id-expression 'i'.
    return std::forward<Self>(self).i;
  }

  template<typename Self>
  auto&& get1(this Self&& self) {
    // P00847R7: mitigate against shadowing by copy-cvref.
    return ((__copy_cvref(Self, B1)&&)self).i;
  }

  template<typename Self>
  auto&& get2(this Self&& self : B1) {
    // Circle deduced forward reference uses a normal forward.
    return std::forward<Self>(self).i;
  }

  int i;
};

struct B2 {
  int i;
};

struct D : B1, B2 { };

int main() {
  D d;

  // Uncomment this for ambiguous declaration error.
  // int x0 = d.get0();

  // Works with explicit upcast to B1.
  int x1 = d.get1();

  // Works with deduced forward reference.
  int x2 = d.get2();
}
