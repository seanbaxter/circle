#include "variant.hxx"
#include <iostream>

using namespace circle;

template<typename... Vars>
using variant_union = variant<
  type...(for typename T : Vars => T.type_args...).unique...
>;

template<typename... Vars>
variant_union<Vars...> 
variant_select(int index, Vars&&... vars forward : variant) {
  using ret_type = variant_union<Vars...>;
  return int... == index ...? ret_type(vars) : __builtin_unreachable();
}

int main() {
  variant<int, double, char> v1 = 'x';
  variant<const char*, float, double, char, short>  v2 = 19i16;

  // If true, select v1. If false, select v2.
  variant v = variant_select(true, v1, v2);

  // Prints 'x'
  visit([](const auto& x) { std::cout<< x<< "\n"; }, v);
}