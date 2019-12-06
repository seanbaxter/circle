#include "format.hxx"

int main() {
  double x = M_PI / 2;

  // Create a syntax error in a format specifier.
  "x = {:10.6r:x}\n"_print;

  return 0;
}