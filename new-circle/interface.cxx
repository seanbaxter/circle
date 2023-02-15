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

  // It looks like we're calling member functions on builtin types!
  // These are interface method calls. Name lookup finds them, because
  // we brought their impls into scope.
  int x = 101;
  double y = 1.618;
  x.print();
  y.print();
}