#include "format.hxx"

struct foo_t;

int main() {
  // Try to print a pointer-to-member function. It's not supported!
  int(foo_t::*pmf)(double, int) = nullptr;

  // Get a nice error here.
  "pointer-to-member pmf = {pmf}\n"_print;

  return 0;
}