#include <cstdio>

void circle_enhanced_bindings() {
  // Declare a designated binding. This binds according to member name
  // instead of position within an aggregate. The names do not have to be
  // ordered according to the data member declarations.
  struct vec4_t {
    int x, y, z, w;
  };
  vec4_t obj { 100, 200, 300, 400 };

  // Bind only the .x and .z components using designated bindings.
  auto& [.x : x1, .z : z1] = obj;
  printf("x1 = %d, z1 = %d\n", x1, z1);

  // Bind only the .x and .z components using wildcards.
  auto& [x2, _, z2, _] = obj;
  printf("x2 = %d, z2 = %d\n", x2, z2);

  // Declare a recursive structured-binding pattern to decompose a 2D
  // array. This is not allowed by C++17, because it only accepts 
  // identifier-list bindings.
  int array[][3] {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  auto& [ [m11, m12, m13], [m21, m22, m23], [m31, m32, m33] ] = array;
  printf("matrix = <%d, %d, %d>, <%d, %d, %d>, <%d, %d, %d>\n", 
    m11, m12, m13, m21, m22, m23, m31, m32, m33);

  // Use both structured and designated bindings to extract the .w 
  // members from each vector.
  vec4_t vecs[] {
    { 10, 11, 12, 13 },
    { 20, 21, 22, 23 },
    { 30, 31, 32, 33 }
  };
  auto& [ [.w : w1], [.w : w2], [.w : w3] ] = vecs;
  printf("w1 = %d, w2 = %d, w3 = %d\n", w1, w2, w3);
}

int main() {
  circle_enhanced_bindings();
  return 0;
}