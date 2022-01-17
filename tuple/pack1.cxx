#include <iostream>

template<typename... Ts>
struct tuple {
  Ts ...m;

  tuple() = default;

  // Converting constructor.
  template<typename... Ts2>
  tuple(Ts2&&... x) : m(x)... { }

  template<int I>
  Ts...[I]& get() {
    return m...[I];
  }
};

int main() {
  tuple<int, double, const char*> x(10, 3.14, "Hello tuple");
  std::cout<< x.get<0>()<< " "<< x.get<1>()<< " "<< x.get<2>()<< "\n";
}