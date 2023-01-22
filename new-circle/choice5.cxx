#pragma feature choice
#include <type_traits>
#include <iostream>

choice Foo {
  x(int),
  y(double),
  z(const char*),
};

template<typename T> requires (T~is_enum)
const char* enum_to_string(T e) noexcept {
  return T~enum_values == e ...? 
    T~enum_names :
    "unknown enum of type {}".format(T~string);
}

int main() {
  // "alternatives" is an enum member of Foo.
  static_assert(Foo::alternatives~is_enum);

  // It has enumerators for each choice alternative.
  std::cout<< "alternatives enumerators:\n";
  std::cout<< "  {} = {}\n".format(Foo::alternatives~enum_names,
    Foo::alternatives~enum_values~to_underlying) ...;

  // Naming a choice alternative gives you back an enumerator.
  static_assert(Foo::alternatives == decltype(Foo::x));

  // Foo::y is an enumerator of type Foo::alternatives. But it's also 
  // how you construct choice types! The enumerator has been overloaded to 
  // work as an initializer. 

  // Foo::y is type Foo::alternatives.
  static_assert(Foo::alternatives == decltype(Foo::y));

  // Foo::y() is an initializer, so Foo::y() is type Foo. 
  static_assert(Foo == decltype(Foo::y()));

  // Initialize a Foo object.
  Foo obj = .y(3.14);

  // .active is an implicit data member set to the active alternative.
  // The type is Foo::alternatives.
  std::cout<< "obj.active = "<< enum_to_string(obj.active)<< "\n";

  // Compare an enumerator with the .active member to see what's active.
  if(Foo::x == obj.active)
    std::cout<< "x is active\n";
  else if(Foo::y == obj.active)
    std::cout<< "y is active\n";
  else if(Foo::z == obj.active)
    std::cout<< "z is active\n";
}