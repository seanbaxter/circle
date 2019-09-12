#include <cstdio>

int sq(int x) {
  return x * x;
}

int main() {
  struct foo_t {
    int x, y, z;
  };
  foo_t obj { 3, 4, 5 };

  // Use / to evaluate an expression test. The _ token inside any pattern test
  // gives the pattern initializer at that point.
  @match(obj) {
    // Compare .z to expressions of _x and _y.
    [_x, _y, sq(_x) + sq(_y)]                   => printf("Sum of squares!\n");
    [_x, _y, abs(sq(_x) - sq(_y))]              => printf("Difference of squares!\n");
    
    // We can bind _z to .z and use a guard
    // [_x, _y, _z] if(sq(_x) + sq(_y) == sq(_z))  => printf("Perfect squares!\n");

    // or we can use / to introduce an expression test. The _ declaration in
    // a pattern test refers to the initializer for that element, in this case
    // .z. We can optionally bind the .z member after the pattern test.
    [_x, _y, / sq(_x) + sq(_y) == sq(_)]        => printf("Perfect squares!\n");

    _                                           => printf("I got nothing.\n");
  };
  return 0;
} 
