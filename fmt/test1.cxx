#include "format.hxx"

int main() {
  int x = 50;
  double y = 100 * M_PI;
  const char* z = "Hello";

  // Basic printing.
  "x = {x}, y = {y}, z = {z}\n"_print;

  // Print as hex.
  "hex x = {:x:x}, y = {:a:y}\n"_print;

  // Use width and precision specifiers. Put the format specifier between
  // : and :.
  "y = {:10.5f:y} or y = {:.10e:y}\n"_print;

  // Center-align the values and fill unused space with ~.
  "x = {:~^15:x}, y = {:~^15:y}, z = {:~^15:z} \n"_print;

  // Provide dynamic width and precision specifiers. These are their own
  // expressions that are evaluated and must yield integral values.
  int width = 15;
  int prec = 7;
  "y with dynamic width/prec = {:{width}.{prec}:y}\n"_print;

  return 0;
}