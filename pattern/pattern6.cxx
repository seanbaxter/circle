#include <cstdio>

int main() {
  struct foo_t {
    int x;
    long y;
    float z;
  };
  foo_t obj { 4, 5, 6 };

  double x = 2 + @match(obj) -> float {
    // Implicitly cast each return expression to float.
    [_x, _y, 5] => _x;    // If z == 5, return x.
    // [_, 5, _z] => _z;    // If y == 5, return z.
    // [5, _y, _] => _y;    // If x == 5, return y.
    _          => 0;     // Else, return 0.
  } / 3;
  
  printf("%f\n", x);
  return 0;
}