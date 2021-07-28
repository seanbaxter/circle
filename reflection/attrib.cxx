#include <cstdio>

using alt_name [[attribute]] = const char*;

template<typename enum_t>
const char* enum_to_string(enum_t e) {
  switch(e) {
    @meta for enum(enum_t e2 : enum_t) {
      case e2:
        if constexpr(@enum_has_attribute(e2, alt_name))
          return @enum_attribute(e2, alt_name);
        else
          return e2.string;
    }

    default:
      return "unknown";
  }
}

enum class shapes_t {
  circle,
  ellipse [[.alt_name="A squishy circle"]],
  square,
  rectangle [[.alt_name="A pancaked square"]],
};

int main() {
  printf("%s\n", enum_to_string(@enum_values(shapes_t)))...;
}