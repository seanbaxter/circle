#include <concepts>
#include <string_view>
#include <type_traits>
#include <iostream>

// Any even test. You must use a constraint so that errors show up as
// substitution failures.
constexpr bool even(std::integral auto x) {
  return 0 == (x % 2);
}

template<typename type_t>
void f(const type_t& x) {
  std::cout<< type_t.string + ": ";
  
  inspect(x) {
    i is int || i is long || i is long long {
      // Use an inspect-group { }.
      is even => std::cout<< "even signed integral "<< i<< "\n";
      is _    => std::cout<< "odd signed integral "<< i<< "\n";
    }

    i is unsigned || i is unsigned long || i is unsigned long long {
      is even => std::cout<< "even unsigned integral "<< i<< "\n";
      is _    => std::cout<< "odd unsigned integral "<< i<< "\n";
    }

    i is even                     => std::cout<< "even "<< i<< "\n";

    is bool                       => {
      // Use a compound statement as the result.
      if(x) std::cout<< "true!\n";
      else  std::cout<< "false!\n";
    }

    // Any object that compares equal with nullptr. This will be nullptr_t
    // or any null-valued pointer type.
    is nullptr                    => std::cout<< "nullptr\n";

    // Anything convertible to string.
    s as std::string              => std::cout<< "string "<< s<< "\n";

    i is std::integral            => std::cout<< "integral "<< i<< "\n";

    f is std::is_floating_point_v => std::cout<< "floating point "<< f<< "\n";

    is _                          => std::cout<< "unsupported type\n";
  }
}

int main() {
  f(501ll);    // long long: odd signed integral 501
  f(400u);     // unsigned: even unsigned integral 400
  f('4');      // char: even 4
  f(19i16);    // short: integral 19
  f(true);     // bool: true!
  f(19.1l);    // long double: floating point 19.1
  f("Yo");     // const char*: string Yo
  f(nullptr);  // std::nullptr_t: nullptr
}