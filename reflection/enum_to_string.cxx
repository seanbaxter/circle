#include <cstdio>
#include <cstring>
#include <optional>

template<typename enum_t>
std::optional<const char*> enum_to_string(enum_t e) {
  switch(e) {
    @meta for enum(enum_t e2 : enum_t) {
      case e2:
        return e2.string;
    }

    default:
      return { };
  }
}

template<typename enum_t>
std::optional<enum_t> string_to_enum(const char* s) {
  @meta for enum(enum_t e : enum_t) {
    if(0 == strcmp(e.string, s))
      return e;
  }
  return { };
}

enum class shapes_t {
  circle,
  ellipse,
  square,
  rectangle,
  octagon,
  trapezoid,
  rhombus,
};

int main(int argc, char** argv) {
  printf("Map from enum values to strings:\n");
  int values[] { 4, 2, 8, -3, 6 };
  for(int x : values) {
    if(auto name = enum_to_string((shapes_t)x)) {
      printf("  Matched shapes_t (%d) = %s\n", x, *name);
    } else {
      printf("  Cannot match shapes_t (%d)\n", x);
    }
  }

  printf("\nMap from strings to enum values:\n");
  const char* names[] {
    "trapezoid", "giraffe", "duck", "circle", "Square", "square"
  };
  for(const char* s : names) {
    if(auto value = string_to_enum<shapes_t>(s)) {
      printf("  Matched shapes_t (%s) = %d\n", s, *value);
    } else {
      printf("  Cannot match shapes_t (%s)\n", s);
    }
  }
}