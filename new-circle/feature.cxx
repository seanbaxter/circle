#include <iostream>

// interface and impl are identifiers.
int interface = 0;
int impl = 1;

// Bring interface, impl, dyn, make_dyn and self in as keywords.
#pragma feature interface self

// interface is a keyword. Define an interface that allows implicit impls.
interface IPrint auto {
  void print() const default (Self~is_arithmetic) {
    std::cout<< self<< "\n";
  }
};

int main() {
  // Bring impl<int, IPrint> and impl<double, IPrint> into scope.
  using impl int, double : IPrint;

  // Call the interface method for impl<double, IPrint>.
  int x = 101;
  double y = 1.618;

  // It looks like we're calling member functions on builtin types!
  // These are interface method calls. Name lookup finds them, because
  // we brought their impls into scope.
  x.print();
  y.print();

  // We can still access the interface and impl object declarations from 
  // the top, using backtick identifiers.
  `interface` = 2;
  `impl` = 3;
}