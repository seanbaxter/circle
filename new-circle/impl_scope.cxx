#pragma feature interface self
#include <iostream>

interface IPrint {
  void print() const;
};

template<typename T> requires(T~is_arithmetic)
impl T : IPrint {
  void print() const {
    std::cout<< T~string + ": "<< self<< "\n";
  }
};

int main() {
  // Put these five impls in scope.
  using impl short, int, long, float, double : IPrint;

  // Because their impls are in scope, we can use
  // unqualified member access to call IPrint::print.
  (1i16).print();
  (2).print();
  (3l).print();
  (4.4f).print();
  (5.55).print();

  // Error: 'print' is not a member of type unsigned.
  // (6u).print();
}
