#feature on interface self
#include <iostream>

// IGroup inherits a pack of interfaces.
// It's marked 'auto', meaning that if a type implements its
// requirements, an implicit impl is generated for it.
// Since it has no interface methods, the only requirements are that
// the type implement all base interfaces IFaces.
template<interface... IFaces>
interface IGroup auto : IFaces... { };

interface IPrint {
  void print() const;
};

interface IScale {
  void scale(double x);
};

interface IUnimplemented { };

template<interface IFace, typename T : IFace>
void func(T& obj) {
  obj.print();
  obj.scale(1.1);
  obj.print();
}

impl double : IPrint {
  void print() const { }
};

impl double : IScale {
  void scale(double x) { }
};

int main() {
  double x = 100;
  func<IGroup<IPrint, IScale>>(x);

  // Error: double does not implement interface IUnimplemented
  func<IGroup<IPrint, IScale, IUnimplemented>>(x);
}