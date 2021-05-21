#include <cstdio>

template <typename T>
struct optional {
  template <typename Self>
  constexpr auto operator->(this Self&& self) {
    return &self.inner;
  }

  T inner;
};

struct obj_t {
  int x, y, z;
};

int main() {
  optional<obj_t> opt;
  opt->x = 1;

  printf("%d\n", opt->x);
}