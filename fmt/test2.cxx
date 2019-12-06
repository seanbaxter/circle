#include "format.hxx"

int main() {
  // The inner parts are opened up recursively until we hit something that's
  // not array-like, and the format specifiers apply to these elements.
  std::vector<float> x { 
    5 * M_PI, 
    M_PI, 
    M_PI / 5 
  };

  // Print the array with default settings.
  "Default settings = {x}\n"_print;

  // Print the array in hexadecimal scientific notation.
  "Hexadecimal array = {:a:x}\n"_print;

  // Print the array with width.precision specifiers. Center-justify the
  // numbers and fill the 11-character width with '*' characters.
  "Fill and width/prec = {:*^11.2:x}\n"_print;

  // Print the array with dynamic width.precision specifiers.
  int y = 9;
  int z = 12;
  "Dynamic width/prec = {:{y + 5}.{z / 4}:x}\n"_print;

  // Initialize a linked list from the vector.
  std::list<double> w {
    x.begin(), x.end()
  };

  // Circle format prints linked lists like other array-like containrs.
  "An std::list<double> = {w}\n"_print;

  return 0;
}
