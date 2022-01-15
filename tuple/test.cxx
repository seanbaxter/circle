#include <iostream>
#include "tuple.hxx"

int main() {
  using namespace circle;

  auto a = make_tuple(100, "Hello tuple", 'x');
  auto b = make_tuple(21.1f, nullptr, 19i16);
  auto c = make_tuple(true, 3.14l);

  auto cat = tuple_cat(std::move(a), std::move(b), c);

  // Print the index, the type of each element, and its value.
  std::cout<< 
    int...<< ": "<< 
    decltype(cat).tuple_elements.string << " is '"<<
    cat...[:]<< "'\n" ...;
}