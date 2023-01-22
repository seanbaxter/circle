#include <type_traits>

template<typename... Ts>
void func() {
  static_assert(
    Ts~is_arithmetic, 
    "parameter {0}, type {1}, is not arithmetic".format(int..., Ts~string)
  ) ...;
}

int main() {
  func<int, double, char*, float>();
}