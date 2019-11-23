#include <cstdio>
#include <string>
#include <iostream>

int main() {
  struct foo_t {
    int x, y;
  };
  auto& [a, b] = foo_t { 1, 2 };

  printf("%s\n", @type_string(decltype(a)));

  return 0;

}