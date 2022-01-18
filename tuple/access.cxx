#include "tuple.hxx"
#include <array>
#include <iostream>

int main() {
  circle::tuple<int, double, const char*> x1(
    100, 3.14, "Hello circle::tuple"
  );

  // Print out by subscript.
  std::cout<< "Print by subscript:\n";
  std::cout<< "  0: "<< x1.[0]<< "\n";
  std::cout<< "  1: "<< x1.[1]<< "\n";
  std::cout<< "  2: "<< x1.[2]<< "\n";

  std::tuple<short, float, std::string> x2(
    50, 1.618f, "Hello std::tuple"
  );

  // Print out by slice.
  std::cout<< "Print by slice:\n";
  std::cout<< "  " + int....string + ": "<< x2.[:]<< "\n" ...;

  std::pair<const char*, long> x3(
    "A pair's string",
    42
  );
  std::cout<< "Works with pairs:\n";
  std::cout<< "  " + int....string + ": "<< x3.[:]<< "\n" ...;

  std::cout<< "Even works with builtin arrays:\n";
  int primes[] { 2, 3, 5, 7, 11 };
  std::cout<< "  " + int....string + ": "<< primes.[:]<< "\n" ...;
}