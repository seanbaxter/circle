#include <tuple>
#include <iostream>

template<typename T>
void print_type() {
  std::cout<< T.string<< "\n";
  std::cout<< T.template.string<< "\n";
  std::cout<< int...<< ": "<< T.type_args.string<< "\n" ...;
}

int main() {
  print_type<std::tuple<char*, double, int, void*>>();
}