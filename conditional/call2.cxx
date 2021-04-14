#include <tuple>
#include <iostream>

template<typename func_t, typename... types_t>
auto call_tuple1(func_t f, const std::tuple<types_t...>& tuple, int index) {
  switch(index) {
    @meta for(int i : sizeof...(types_t)) {
      case i:
        return f(tuple...[i]);
    }
  }
}

template<typename func_t, typename... types_t>
void call_tuple2(func_t f, const std::tuple<types_t...>& tuple, int index) {
  // Like the above, but one line.
  return int... == index ...? f(tuple...[:]) : __builtin_unreachable();
}

int main() {
  auto f = [](const auto& x) {
    std::cout<< x<< "\n";
  };

  auto tuple = std::make_tuple(1, 5.5, "Hello tuple");

  call_tuple1(f, tuple, 0);
  call_tuple2(f, tuple, 1);
  call_tuple2(f, tuple, 2);
}