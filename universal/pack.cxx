#include <tuple>
#include <iostream>

int main() {
  auto tuple = std::make_tuple('a', 2, 3.3);
  std::cout<< int...<< ": "<< tuple.[:]<< "\n" ...;
}