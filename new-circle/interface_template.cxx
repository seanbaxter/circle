#pragma feature interface self
#include <iostream>

interface IPrint {
  void print() const;
};

interface IScale {
  void scale(double x);
};

template<typename T : IPrint & IScale>
void func(T& obj) {
  obj.print();
  obj.scale(1.1);
  obj.print();
}

impl double : IPrint {
  void print() const {
    std::cout<< self<< "\n";
  }
};

impl double : IScale {
  void scale(double x) {
    self *= x;
  }
};

int main() {
  double x = 100;
  func(x);

  int y = 101;

  // Error: int does not implement interface IPrint.
  // func(y);
}