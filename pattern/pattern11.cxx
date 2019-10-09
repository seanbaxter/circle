#include <tuple>
#include <cstdio>

auto dot_product(auto a, auto b) {
  auto& [...p1] = a;
  auto& [...p2] = b;
  return (... + (p1 * p2));
}

template<typename... types_t>
struct tuple_t {
  @meta for(size_t i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};

int main() {
  // std::pair registers as a tuple.
  double x1 = dot_product(std::make_pair(2.1, 3), std::make_pair(4, 2l));
  printf("x1 = %f\n", x1);

  // std::tuple
  double x2 = dot_product(std::make_tuple(1.1, 3.1f), std::make_tuple(2, 3));
  printf("x2 = %f\n", x2);

  // tuple_t undergoes structured binding as a normal class.
  tuple_t<double, float, int> v1 { 1.5, 2.1f, 4 }, v2 { 2.2, 1.3f, 9 };
  double x3 = dot_product(v1, v2);
  printf("x3 = %f\n", x3);

  // Use @member_pack to apply fold directly.
  double x4 = (... + (@member_pack(v1) * @member_pack(v2)));
  printf("x4 = %f\n", x4);

  return 0;
}