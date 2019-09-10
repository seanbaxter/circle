#include <cstdio>

int main() {

  struct foo_t {
    int x, y, z;
  };
  foo_t obj { 3, 4, 5 };

  int Z = 6;
  @match(obj) {
    // Test an expression against the initializer.
    [_, _, 3]    => printf(".z is 3\n");  // structured binding
    [  .z: 4]    => printf(".z is 4\n");  // designated binding

    // Is Z a test/expression or a binding? If the clause fails, it's got to
    // be a test.
    [_, _, Z]    => printf("Z must be a binding\n");
    _            => printf("Z must be an expression\n");
  };

  return 0;
}