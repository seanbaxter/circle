// P2392R1 2.1.2
#include <iostream>
#include <utility>
#include <concepts>

template<typename A, typename B>
std::ostream& operator<<(std::ostream& os, const std::pair<A, B>& pair) {
  os<< "{ "<< pair.first<< ", "<< pair.second<< " }";
  return os;
}

auto in(const auto& min, const auto& max) {
  // The constraint on the parameter x is requried to prevent
  // an ill-formed definition. A modern C++ idiom is to write the
  // expression three times:
  return [=](const auto& x) 
    noexcept(noexcept(x <= min && x <= max))
    requires(requires{ x <= min && x <= max; }) {
    return min <= x && x <= max;
  };
}

void test(auto x) {
  inspect(x) {
    is 3                   => std::cout<< x<< " is 3\n";
    is in(1, 2)            => std::cout<< x<< " is in(1, 2)\n";
    if (2 < x && x <= 3)   => std::cout<< x<< " is in(2, 3)\n";
    is std::pair<int, int> => std::cout<< x<< " is std::pair<int, int>\n";
    is std::pair           => std::cout<< x<< " is std::pair\n";
    is std::integral       => std::cout<< x<< " is std::integral\n";
  }
}

int main() {
  test(3);
  test(1.5);
  test(2.5f);
  test(std::make_pair(1, 2));
  test(std::make_pair(1.1f, 2.2f));
  test(100);
}
