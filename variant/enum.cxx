#include <utility>
#include <iostream>

enum shapes_t {
  circle, square, rhombus = 20, trapezoid, triangle, ellipse
};

template<int I, int J, shapes_t z>
void f() {
  std::cout<< I<< " "<< J<< " "<< z.string<< "\n";
}

int main() {

  // Invoke with an element from a interval.
  constexpr int XDim = 10;
  int x = 3;

  // Invoke with an element from a collection.
  using YSequence = std::index_sequence<1, 3, 5, 7, 9>;
  int y = 7;

  // Invoke with an element from an enumeration.
  shapes_t z = rhombus;

  __visit<XDim, YSequence, shapes_t>(
    f<indices...>(),
    x, y, z
  );
}