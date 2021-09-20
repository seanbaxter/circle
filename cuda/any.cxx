#include <type_traits>
#include <any>
#include <variant>
#include <string>
#include <cuda_runtime.h>
#include <cstdio>

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

enum class shapes_t {
  circle, square, trapezoid, rhombus
};

template<typename enum_t>
const char* enum_to_string(enum_t e) {
  return e == enum_t.enum_values ...? 
    enum_t.enum_names : "unknown " + enum_t.string;
}

template<typename type_t>
void match(const type_t& x) {
  inspect(x) {
    s as shapes_t    => printf("  shapes_t: %s\n", enum_to_string(s));
    i as int         => printf("  int: %d\n", i);
    f as float       => printf("  float: %f\n", f);
    d as double      => printf("  double: %f\n", d);
    s as const char* => printf("  const char*: '%s'\n", s);
    s as std::string => printf("  std::string: '%s'\n", s.c_str());
      is _           => printf("  unknown contents\n");
  }
}

__global__ void kernel() {
  // Pattern match on std::any.
  printf("Pattern matching on std::any:\n");
  match(std::any(5));                 
  match(std::any("An any C string")); 
  match(std::any(3.14));              

  // Pattern match on std::variant.
  printf("\nPattern matching on std::variant:\n");
  typedef std::variant<float, shapes_t, const char*, std::string> var;
  
  match(var(std::string("A variant std::string")));
  match(var(1.618f));
  match(var(shapes_t::trapezoid));
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}