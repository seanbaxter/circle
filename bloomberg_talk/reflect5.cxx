#include <cstdio>
#include <cstring>
#include <optional>

enum class shape_t {
  circle, 
  square,
  triangle,
  hexagon,
};

template<typename type_t>
const char* enum_to_name(type_t e) {
  switch(e) {
    @meta for enum(type_t e2 : type_t) {
      case e2:
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

template<typename type_t>
std::optional<type_t> name_to_enum(const char* name) {
  @meta for enum(type_t e2 : type_t) {
    if(!strcmp(@enum_name(e2), name))
      return e2;
  }
  return { };
}

int main() {
  shape_t shapes[] { 
    shape_t::circle, (shape_t)6, shape_t::hexagon
  };
  const char* names[] {
    "square", "triangle", "rhombus"
  };

  printf("enum_to_name:\n");
  for(shape_t s : shapes) {
    if(const char* name = enum_to_name(s))
      printf("  %s\n", name);
    else
      printf("  <unknown>\n");
  }

  printf("name_to_enum:\n");
  for(const char* name : names) {
    if(std::optional<shape_t> e = name_to_enum<shape_t>(name)) {
      printf("  %d\n", (int)*e);
    } else
      printf("  <unknown>\n");
  }

  return 0;
}