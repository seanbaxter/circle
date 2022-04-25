#include "variant.hxx"
#include <iostream>

using namespace circle;   // for circle::variant

template<typename T1, typename T2, typename... Ts1, typename... Ts2>
variant<Ts1.union(Ts2...)...> variant_select(
  bool b,
  T1&& v1 forward : variant<Ts1...>,
  T2&& v2 forward : variant<Ts2...>
) {
  using ret_type = variant<Ts1.union(Ts2...)...>;
  return b ? ret_type(v1) : ret_type(v2);
}

int main() {  
  variant<int, double, char> v1 = 'x';
  variant<const char*, float, double, char, short>  v2 = 19i16;

  // If true, select v1. If false, select v2.
  auto v = variant_select(true, v1, v2);

  std::cout<< decltype(v).string + "\n";
  visit([](const auto& x) { std::cout<< x<< "\n"; }, v);
}
