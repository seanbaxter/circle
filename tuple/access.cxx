#include <tuple>
#include <array>
#include <iostream>

int main() {
  std::tuple<int, double, const char*> tup(
    100, 3.14, "Hello std::tuple"
  );

  std::cout<< int...<<": "<< decltype(tup).tuple_elements.string<< "\n" ...;

  // Print out by subscript.
  std::cout<< "Print by subscript:\n";
  std::cout<< "  0: "<< tup.[0]<< "\n";
  std::cout<< "  1: "<< tup.[1]<< "\n";
  std::cout<< "  2: "<< tup.[2]<< "\n";

  // Print out by slice.
  std::cout<< "Print by slice - "<< sizeof. tup<< " elements:\n";
  std::cout<< "  "<< int...<< ": "<< tup.[:]<< "\n" ...;

  std::pair<const char*, long> pair(
    "A pair's string",
    42
  );
  std::cout<< "Works with pairs - "<< sizeof. pair<< " elements:\n";
  std::cout<< "  "<< int...<< ": "<< pair.[:]<< "\n" ...;

  int primes[] { 2, 3, 5, 7, 11 };
  std::cout<< "Works with builtin arrays - "<< sizeof. primes<< " elements:\n";
  std::cout<< "  "<< int...<< ": "<< primes.[:]<< "\n" ...;
}