#include <iostream>
#include <type_traits>

template<typename type_t>
const char* name_from_enum(type_t e) {
  static_assert(std::is_enum<type_t>::value);

  switch(e) {
    // A ranged-for over the enumerators in this enum.
    @meta for enum(type_t e2 : type_t) {
      // e2 is known at compile time, so we can use it in a case-statement.
      case e2:
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

int main() {

  enum color_t {
    red, blue, green, yellow, purple, violet, chartreuse, puce,
  };
  color_t colors[] { yellow, violet, puce };

  // Print all the color names in the array.
  for(color_t c : colors)
    std::cout<< name_from_enum(c)<< "\n";

  return 0;
}