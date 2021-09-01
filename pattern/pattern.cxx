#include <concepts>
#include <iostream>
#include <tuple>

template<class T, bool V = requires { sizeof(std::tuple_size<T>); }> 
constexpr bool is_tuple_like = V;

// Any even test. You must use a constraint so that errors show up as
// substitution failures.
constexpr bool even(std::integral auto x) {
  return 0 == (x % 2);
}

template<typename T> requires(std::is_class_v<T>)
std::ostream& operator<<(std::ostream& os, const T& obj) {
  if constexpr(is_tuple_like<T>) {
    // Bust the tuple into components and print.
    os<< "[";
    os<< (int... ? ", " : " ")<< obj...[:] ...;
    os<< " ]";

  } else {
    // Use reflection and print non-static public data members.
    os<< "{";
    os<< (int... ? ", " : " ")<< T.member_names<< ":"<< obj...[:] ...;
    os<< " }";
  }
  return os; 
}

template<typename type_t>
void f(const type_t& x) {
  using std::integral, std::cout;

  cout<< x<< ": ";
  inspect(x) {
    [a, b] is [int, int]           => cout<< "2-int tuple "<< a<< " "<< b<< "\n";
    [_, y] is [0, even]            => cout<< "point on y-axis and even y "<< y<< "\n";
    [a, b] is [integral, integral] => cout<< "2-integral tuple "<< a<< " "<< b<< "\n";
    [...pack]                      => (cout<< pack<< " " ...)<< "\n";
    is _                           => cout<< "((no matching value))\n";
  }
}

int main() {
  // Run tuple-like things through the pattern matcher.
  f(std::make_tuple(4, 5));
  f(std::make_pair(0u, 10u));
  f(std::make_pair(5, 10ll));
  f(std::make_tuple(1, 2, 3, 4, 5));

  // Run classes through the pattern matcher.
  struct foo_t {
    long x, y;
  };
  f(foo_t { 10, 16 });

  struct bar_t {
    double a, b, c, d;
  };
  f(bar_t { 5.5, 6.6, 7.7, 8.8 });
}