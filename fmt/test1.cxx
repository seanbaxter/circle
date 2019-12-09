#include "format.hxx"

int main() {
  int x = 50;
  double y = 100 * M_PI;
  const char* z = "Hello";

  // Basic printing.
  "x = {x}, y = {y}, z = {z}\n"_print;

  // Print as hex.
  "hex x = {x:x}, y = {y:a}\n"_print;

  // Use width and precision specifiers. Put the format specifier between
  // : and :.
  "y = {y:10.5f} or y = {y:.10e}\n"_print;

  // Center-align the values and fill unused space with ~.
  "x = {x:~^15}, y = {y:~^15}, z = {z:~^15} \n"_print;

  // Provide dynamic width and precision specifiers. These are their own
  // expressions that are evaluated and must yield integral values.
  int width = 15;
  int prec = 7;
  "y with dynamic width/prec = {y:{width}.{prec}}\n"_print;

  return 0;
}
