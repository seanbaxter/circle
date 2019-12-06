#include "format.hxx"

struct vec3_t {
  float x, y, z;

  friend vec3_t operator*(float a, vec3_t v) {
    return { a * v.x, a * v.y, a * v.z };
  }
};

template<typename type_t>
type_t sq(type_t x) {
  return x * x;
}

int main() {
  float a = 1.5;
  vec3_t v { 2, 3, 1 };
  vec3_t z = a * v;

  // Automatically print a class object by member.
  "a = {a}\n"_print;
  "v = {v}\n"_print;

  // Allow any expression in the format specifier. Reflection still works.
  "a * v = {a * v}\n"_print;

  // Allow pack expressions inside the format specifier. Here a unary fold
  // operator takes the L2 norm of the vector.
  "|v| = {sqrt((sq(@member_values(v)) + ...))}\n"_print;

  // Use a pack format to print just the member names as a collection.
  "v has member names {...@member_names(vec3_t)}\n"_print;

  // Or do it on the member values.
  "v has member values {...@member_values(v)}\n"_print;

  // Circle recognizes STL types and handles them differently from generic
  // class objects.
  std::optional<vec3_t> val1, val2;
  val2 = v;

  "val1 = {val1}, val2 = {val2}\n"_print;

  return 0;
}