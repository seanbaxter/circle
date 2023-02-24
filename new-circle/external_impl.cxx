#feature on interface self
#include <iostream>

interface IPrint {
  void print() const;
};

impl double : IPrint {
  void print() const;
};

template<typename T> requires(T~is_arithmetic)
impl T : IPrint {
  void print() const;
};

// Out-of-line definition for impl<double, IPrint>::print.
// This has external linkage.
void impl<double, IPrint>::print() const {
  std::cout<< "explicit specialization: "<< self<< "\n";
}

// Out-of-line definition for impl<T, IPrint>::print. 
// This has inline linkage, because it's a template entity.
template<typename T> requires (T~is_arithmetic)
void impl<T, IPrint>::print() const {
  std::cout<< "partial template: "<< self<< "\n";
}

int main() {
  (3.14).IPrint::print();  // Calls the explicit specialization.
  (101ul).IPrint::print(); // Calls the partial specialization.
}

