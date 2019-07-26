#include <iostream>

template<typename... params_t>
void func(const params_t&... params) {
  // Print each of the arguments.
  std::cout<< params<< "\n" ...;
}

struct foo_t {
  int x;
  double y;
  std::string z;
};

int main() {
  foo_t foo { 1, 3.14, "Hello" };

  // Expand the foo_t instance into a list of its members.
  func(@member_pack(foo)...);

  return 0;
}