#include <cstdio>

using name [[attribute]] = const char*;
using width [[attribute]] = int;

template<typename type_t>
void func() {

  // This works fine.
  printf("@enum_has_attribute:\n");
  @meta for(int i = 0; i < @enum_count(type_t); ++i){ 
    if constexpr(@enum_has_attribute(type_t, i, name)) {
      printf("  %s : %s\n", @enum_name(type_t, i),
        @enum_attribute(type_t, i, name));
    }
  }

  // This always misses the attributes on the enumerators of type_t.
  // Why? Because we're literally asking for attributes on the loop object 
  // e, which is a different declaration from the underlying enumerator it
  // gets set to each step.
  printf("Loop with @has_attribute:\n");
  @meta for enum(type_t e : type_t) {
    if constexpr(@has_attribute(e, name)) {
      printf("  %s : %s\n", e.string, @attribute(e, name));
    }
  }

  // This works fine, because @enum_attribute and @enum_has_attribute
  // reflects on attributes of the enumerator with the provided value, rather
  // than reflecting on the provided declaration.
  printf("Loop with @enum_has_attribute:\n");
  @meta for enum(type_t e : type_t) {
    if constexpr(@enum_has_attribute(e, name)) {
      printf("  %s : %s\n", e.string, @enum_attribute(e, name));
    }
  }
}

enum my_enum_t {
  a [[.name="Foo", .width=100]],
  b [[.name="Bub"]],
  c,
};

int main() {
  func<my_enum_t>();
}