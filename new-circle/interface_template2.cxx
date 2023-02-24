#feature on interface self

interface IPrint {
  void print() const;
};

interface IScale {
  void scale(double x);
};

interface IUnimplemented { };

// IFace is an interface pack.
// Expand IFace into the interface-list that constrains T.
template<interface... IFace, typename T : IFace...>
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
  func<IPrint, IScale>(x);

  // Error: double does not implement interface IUnimplemented
  func<IPrint, IScale, IUnimplemented>(x);
}