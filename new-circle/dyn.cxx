#pragma feature interface self
#include <iostream>
#include <string>

interface IFace { 
  void print() const;
};

impl int : IFace { 
  void print() const { 
    std::cout<<"int.print = "<< self<< "\n";
  }
};

impl std::string : IFace {
  void print() const {
    std::cout<<"string.print = "<< self<< "\n";
  }
};

int main() {
  // dyn<IFace> implements IFace.
  static_assert(impl<dyn<IFace>, IFace>);

  // Get a pointer to the type-erased dyn<IFace>.
  int x = 100;
  dyn<IFace>* p1 = make_dyn<IFace>(&x);

  // Call its interface method. Look, it doesn't say 'int' anywhere!
  p1->IFace::print();

  // Type erase string and call IFace::print.
  std::string s = "My nifty string";
  dyn<IFace>* p2 = make_dyn<IFace>(&s);
  p2->IFace::print();
}

