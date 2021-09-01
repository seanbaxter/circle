#include <iostream>
#include <concepts>
#include <variant>
#include <any>

// std::variant operator-is support.
template<typename T, typename... Ts>
requires((... || T is Ts))
constexpr bool operator is(const std::variant<Ts...>& x) {
  return holds_alternative<T>(x);
}

template<typename T, typename... Ts>
requires((... || T is Ts))
constexpr T& operator as(std::variant<Ts...>& x) { 
  return get<T>(x);
}

template<typename T, typename... Ts> 
requires((... || T is Ts))
constexpr const T& operator as(const std::variant<Ts...>& x) { 
  return get<T>(x);
}

// std::any operator-is support.
template<typename T>
constexpr bool operator is(const std::any& x) { 
  return typeid(T) == x.type(); 
}

template<typename T> requires (!T.is_reference)
constexpr T operator as(const std::any& x) {
  return any_cast<T>(x);
}

template<typename T> requires (T.is_reference)
constexpr T& operator as(std::any& x) {
  if(auto p = any_cast<T.remove_reference*>(&x))
    return *p;
  throw std::bad_any_cast();
}

// Any even test. You must use a constraint so that errors show up as
// substitution failures.
constexpr bool even(std::integral auto x) {
  return 0 == (x % 2);
}

void f(const auto& x) {
  inspect(x) {
    s as short          => std::cout<< "short<< "<< s<< "\n";
    i as int            => std::cout<< "int "<< i<< "\n";
    i is std::integral {
      is even => std::cout<< "even non-int integral "<< i<< "\n";
      is _    => std::cout<< "odd non-int integral "<< i<< "\n";
    }
    s as std::string    => std::cout<< "string "<< s<< "\n";
    is _                => std::cout<< "((no matching value))\n";
  }
}

int main() {
  std::variant<float, char, int, long, std::string> var;
  var = 5;
  f(var);

  var = 5l;
  f(var);

  std::any any = std::string("Hello any");
  f(any);

  any = 100;
  f(any);
}
