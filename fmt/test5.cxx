#include "format.hxx"

int main() {
  double x = M_PI / 2;

  // Create a syntax error in a format specifier.
  "x = {x:10.6r}\n"_print;

  return 0;
}