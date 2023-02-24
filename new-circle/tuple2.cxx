#feature on new_decl_syntax tuple
#include <tuple>
#include <iostream>

// [tuple] allows a convenient way to do multiple return values.
fn func(x: int) -> (int, float, double) {
  // Return a tuple of type (int, float, double). The [tuple] feature
  // gives us multiple return value syntax.
  return (x, 2.2f * x, 3.3 * x);
}

fn main() -> int {
  var values := func(10);
  std::cout<< decltype(values)~string + "\n";
  std::cout<< values.[:]<< "\n" ...;  // 10 22 33
}