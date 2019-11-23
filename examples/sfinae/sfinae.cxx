#include <cstdio>

template<typename type_t>
void go(type_t& obj) {
  // Try to set obj.x = 1.
  if constexpr(@sfinae(obj.x = 1)) {
    printf("Setting %s obj.x = 1.\n", @type_string(type_t));
    obj.x = 1;
  }

  // Try to call obj.y().
  if constexpr(@sfinae(obj.y())) {
    obj.y();
  }

  // Try to use type_t::big_endian as a value.
  if constexpr(@sfinae((bool)type_t::big_endian)) {
    printf("%s is big endian.\n", @type_string(type_t));
  }
}

struct a_t {
  int x;
  enum { 
    // Declare a list of flags to detect here. The values of the enums
    // don't matter, only if they are present or not.
    big_endian,
    zext,
  };
};

struct b_t {
  // obj.x = 1 is invalid, so it should fail during @sfinae.
  void x() { }

  void y() { 
    printf("b_t::y() called.\n");
  }
};

int main() {
  a_t a; go(a); 
  b_t b; go(b);
  return 0;
}
