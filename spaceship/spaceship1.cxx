#include <cstdio>
#include <compare>

struct int3_t {
  int x, y, z;

  auto operator<=>(const int3_t& rhs) const = default;
};

int main() {
  int3_t a { 1, 2, 3 }, b { 1, 2, 4 }, c { 1, 1, 5 };
  printf("%s\n", @type_name(decltype(a <=> b)));

  bool x = a < b;
  printf("%d\n", x);

  bool y = a < c;
  printf("%d\n", y);

  bool z = b == c;
  printf("%d\n", z);
  
  return 0;
}