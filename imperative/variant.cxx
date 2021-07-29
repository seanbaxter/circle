#include <variant>
#include <iostream>

// Use argument deduction to access the variant alternatives.
template<typename... Ts>
const char* variant_type_string(const std::variant<Ts...>& v) {
  return int... == v.index() ...? Ts.string : "valueless-by-exception";
}

int main() {
  std::variant<int, float, double, const char*> v;

  v = "Hello";
  std::cout<< variant_type_string(v)<< "\n";

  v = 3.14f;
  std::cout<< variant_type_string(v)<< "\n";

  // Use type traits to access the template arguments with having to
  // use argument deduction.
  v = 100;
  std::cout<< (int... == v.index() ...? 
    decltype(v).type_args.string : "valueless-by-exception")<< "\n";
}