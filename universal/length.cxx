#include <tuple>
#include <utility>
#include <cstdio>

int main() {
  // sizeof... gives the length of a pack structured binding.
  auto [...pack] = std::make_pair(1, 2.0);
  printf("pack.length = %d\n", sizeof... pack);

  // sizeof... gives the number of members in a tuple-like type.
  printf("tuple.length = %d\n", sizeof... std::make_tuple('a', 2, 3.0));

  // sizeof... gives the length of the array.
  int my_array[] { 1, 2, 3, 4, 5, 6 };
  printf("my_array.length = %d\n", sizeof... my_array);

  // sizeof... gives the number of non-static public data members.
  struct obj_t {
    int x, y, z;
    const char* w;
  };
  printf("obj.length = %d\n", sizeof...(obj_t));
}