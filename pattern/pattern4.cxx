#include <cstdio>

int sq(int x) {
  return x * x;
}

int main() {
  struct foo_t {
    int x, y, z;
  };
  foo_t obj { 3, 4, 7 };

  @match(obj) {
    [_x, _y, sq(_x) + sq(_y)]                   => printf("Sum of squares!\n");
    [_x, _y, abs(sq(_x) - sq(_y))]              => printf("Difference of squares!\n");
    [_x, _y, _z] if(sq(_x) + sq(_y) == sq(_z))  => printf("Perfect squares!\n");
    _                                           => printf("I got nothing.\n");
  };
  return 0;

} 