#include <cstdio>
#include <type_traits>

template<typename type_t>
const char* enum_to_name(type_t e) {
  static_assert(std::is_enum<type_t>::value, 
    "enum_to_name must be called on enum type");

  // e is only known at runtime. Generate a switch to find its 
  // compile-time known enumerator.
  switch(e) {
    // Loop over all enumerators in the enum.
    @meta for enum(type_t e2 : type_t) {
      // If the runtime e matches the compile-time e2...
      case e2:
        // Return the string name of e2.
        return @enum_name(e2);
    }

    default:
      // The argument is not an enumerator.
      return "<unknown>";
  }
}

int main() {
  enum my_enum_t {
    a, b, c, d, e
  };

  const char* name = enum_to_name(c);
  printf("enum is %s\n", name);

  return 0;
}