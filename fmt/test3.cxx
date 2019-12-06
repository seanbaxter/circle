#include "format.hxx"

enum class shape_t {
  circle,
  square,
  octagon, 
  triangle,
};

int main() {
  shape_t shapes[] {
    shape_t::square, 
    shape_t::octagon,
    shape_t::triangle,
    (shape_t)27
  };

  // Print the enums in the array with default settings. The enumerator names
  // are printed when available.
  "shapes = {shapes}\n"_print;

  // Center the enum names and use '~' to fill.
  "shapes = {:~^15:shapes}\n"_print;
 
  // Use reflection to print all enum names in a loop.
  "Your enum names are:\n"_print;
  int counter = 0;
  @meta for enum(shape_t e : shape_t)
    "{counter++}: {@enum_name(e)}\n"_print;

  // Print all enum names using format pack expansion. This puts them
  // all in a collection.
  "enum names = {...@enum_names(shape_t)}\n"_print;

  // Use 12-character width and center. This applies to
  // each element in the pack expansion.
  "enum names = {:^12:...@enum_names(shape_t)}\n"_print;

  return 0;
}