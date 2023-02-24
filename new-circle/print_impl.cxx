#feature on interface self
#include <iostream>

interface IScale {
  // Self is dependent in the interface context, and non-dependent
  // in the impl context.
  void scale(Self x);
};

interface IPrint {
  void print() const;
};

// Implement IScale on double. 
impl double : IScale {
  // Self is an alias for double, so can also write this as 
  // void scale(double x)
  void scale(Self x) {
    // There is no implicit object. You must use `this` or `self`
    // to access objects of non-static member functions.
    std::cout<< "impl<double, IScale>::scale(): " << self << " to ";
    self *= x;
    std::cout<< self << "\n";
  }
};

// A partial template that will undergo successfull argument 
// deduction for arithmetic types.
template<typename T> requires(T~is_arithmetic)
impl T : IPrint {
  void print() const {
    std::cout<< "impl<" + T~string + ", IPrint>::print(): "<< self<< "\n";
  }
};

int main() {
  double x = 100;
  x.IScale::scale(2.2);
  x.IPrint::print();
}