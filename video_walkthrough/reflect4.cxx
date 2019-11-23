#include <cstdio>

struct foo_t {
  int x;
  double y;
  float z;
  short w;
};

int main() {
  foo_t obj { 2, 3, 4, 5 };
  double x = (... + @member_values(obj));
  printf("x = %f\n", x);

  return 0;
}