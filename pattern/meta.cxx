#include <type_traits>
#include <iostream>
#include <tuple>

template<class T, bool V = requires { sizeof(std::tuple_size<T>); }> 
constexpr bool is_tuple_like = V;

enum class shapes_t {
  circle, triangle, square, pentagon, hexagon, heptagon, octagon, 
};

template<typename type_t>
std::string to_string(const type_t& x) {
  return inspect(x) -> std::string {
    // You can put meta statements in inspect-definitions.
    @meta std::cout<< "Compiling inspect for "<< type_t.string<< "\n";
    
    // If the argument type is an enum, enter a new inspect-definition.
    is std::is_enum_v {
      // Loop over all enums in type_t and print them.
      @meta for enum(type_t e : type_t)
        is e => e.string;
    }

    is is_tuple_like    => 
      "[" + (... + 
        (int... ?? ", " : "") + to_string(x.[:])
      ) + "]";

    // All tuple-like types must have already returned.
    static_assert(!is_tuple_like<type_t>);
    
    is std::is_class_v  => 
      "{" + (... + 
        (int... ?? ", " : "") + type_t.member_names + ": " + to_string(x.[:])
      ) + "}";

    // All class objects must have already returned.
    static_assert(!type_t.is_class);

    x is std::integral as long long => std::to_string(x);
    x as long double                => std::to_string(x);

    is _             => "unknown value of type " + type_t.string;
  };
}

template<typename T1, typename T2, typename T3>
struct foo_t {
  T1 x;
  T2 y;
  T3 z;
};

int main() {
  std::cout<< to_string(shapes_t::pentagon)<< "\n";
  std::cout<< to_string(100)<< "\n";
  std::cout<< to_string(std::make_pair(shapes_t::triangle, 101))<< "\n";

  foo_t foo {
    shapes_t::octagon, 
    std::make_tuple(1, 2), 
    3.3
  };

  std::cout<< to_string(foo)<< "\n";

  std::cout<< to_string(&foo)<< "\n";
}