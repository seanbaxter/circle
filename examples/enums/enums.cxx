#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <optional>

template<typename type_t>
const char* name_from_enum(type_t x) {
  static_assert(std::is_enum<type_t>::value);
  
  switch(x) {
    @meta for(int i = 0; i < @enum_count(type_t); ++i) {
      // @enum_value is the i'th unique enumerator in type_t.
      // eg, circle, square, rhombus
      case @enum_value(type_t, i):
        // @enum_name returns a string literal of the enumerator.
        return @enum_name(type_t, i);
    }

    default:
      return nullptr;
  }
}

template<typename type_t>
std::optional<type_t> enum_from_name(const char* name) {
  static_assert(std::is_enum<type_t>::value);

  @meta for(int i = 0; i < @enum_count(type_t); ++i) {
    if(0 == strcmp(@enum_name(type_t, i), name))
      return @enum_value(type_t, i);
  }

  return { };
}

enum class shapes_t {
  circle,
  square,
  rhombus,
  nonagon,
  water,
};

int main(int argc, char** argv) {
  if(2 != argc) {
    printf("Usage: enums shape-name\n");
    exit(1);
  }

  if(auto shape = enum_from_name<shapes_t>(argv[1])) {
    printf("%s is a shape.\n", name_from_enum(*shape));

  } else {
    printf("%s is not a shape.\n", argv[1]);
  }

  return 0;
}