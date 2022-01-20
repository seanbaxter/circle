#include <tuple>
#include <iostream>

int main() {
  auto f = [](auto... x) {
    std::cout<< "  "<< int...<< ": "<< x<< "\n" ...;
  };
  auto tup = std::make_tuple(1, 2.2, "Three");

  // Use std::apply to break a tuple into elements and foward to a callable.
  std::cout<< "std::apply:\n";
  std::apply(f, tup);

  // We don't need an apply function for this. The implicit slice operator 
  // does this
  // for us.
  std::cout<< "implicit slice:\n";
  f(tup...);

  // In fact, we don't need a function at all. Use the tuple slice operator
  // .[:] to packify the elements of a tuple and write the operation you
  // want directly.
  std::cout<< "pack expression:\n";
  std::cout<< "  "<< int...<< ": "<< tup.[:]<< "\n" ...;
}