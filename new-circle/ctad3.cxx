#include <vector>
#include <iostream>

template<typename X>
struct foo_t {
  // This constructor provides an implicit deduction guide.
  foo_t(X begin, X end) { }

  template<typename Iter>
  foo_t(Iter it) { }
};

// User-defined deduction guides give you explicit control and are safer.
template<typename X>
foo_t(X x) -> foo_t<typename std::iterator_traits<X>::value_type>;

int main() {
  using vec = std::vector<int>;
  vec v1;

  // Matches an implicit deduction guide.
  foo_t obj1(v1.begin(), v1.end());
  static_assert(foo_t<vec::iterator> == decltype(obj1));

  // Matches the user-defined deduction guide.
  foo_t obj2(v1.begin());
  static_assert(foo_t<int> == decltype(obj2));

  // Break when CTAD chooses an implicit (constructor) deduction guide.
  #pragma feature no_implicit_deduction_guides
  foo_t obj3(v1.begin(), v1.end());

  // Break when CTAD chooses any deduction guide.
  #pragma feature no_class_template_argument_deduction
  foo_t obj4(v1.begin());  

  // Always allow copy-ctor CTAD. That's just convenient.
  foo_t obj5 = obj2;
}