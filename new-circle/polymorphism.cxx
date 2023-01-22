// Classes: methods are bound with data.
struct A1 {
  void print() const;
};

struct B1 {
  void print() const;
};

// Overloading: free functions are overloaded by their receiver type.
struct A2 { };
void print(const A2& a);

struct B2 { };
void print(const B2& b);

// Interfaces: types externally implement interfaces.
// Rather than function overloading, interfaces are implemented by types.
#pragma feature interface

interface IPrint {
  void print() const;
};

struct A3 { };
struct B3 { };

impl A3 : IPrint {
  void print() const;
};

impl B3 : IPrint {
  void print() const;
};

void call() {
  A1 a1;
  a1.print();          // A member function call.

  A2 a2;
  print(a2);           // A free function call.

  A3 a3;
  a3.IPrint::print();  // An interface function call.
}