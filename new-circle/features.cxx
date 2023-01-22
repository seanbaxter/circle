// Enable four features:
// [interface] - Enables the dyn, interface, impl and make_dyn keywordcs.
// [tuple] - Enables new syntax for tuple expressions and types.
// [choice] - Enables the choice and match keywords.
// [self] - Retires 'this' and replaces it with the lvalue 'self'.
#pragma feature interface tuple choice self

// These files are included after the features have been activated, but are
// unaffected by them. Every file on disk gets its own feature mask.
#include <iostream>
#include <tuple>

struct Obj {
  void Inc(int inc) {
    // 'self' is new.
    self.x += inc;

    // Error, 'this' is disabled. That's new.
    // this->x -= inc;
  }
  int x;
};

// Choice types are new.
template<typename T, typename U>
choice Result {
  Ok(T),
  Err(U),
};

void ProcessChoice(Result<Obj, std::string> obj) {
  // Pattern matching is new.
  match(obj) {
    .Ok(auto obj)  => std::cout<< "Got the value "<< obj.x<< "\n";
    .Err(auto err) => std::cout<< "Got the error '"<< err<< "'\n";
  };
}

// Interfaces are new.
interface IPrint { 
  void print() const;
  std::string to_string() const;
};

// Impls are new.
template<typename T> requires (T~is_arithmetic)
impl T : IPrint {
  void print() const {
    std::cout<< T~string + ": "<< self<< "\n";
  }
  std::string to_string() const {
    return T~string + ": " + std::to_string(self);
  }
};

int main() {
  // Create choice objects by naming the active alternative.
  ProcessChoice(.Ok({ 5 }));
  ProcessChoice(.Err("An error string"));

  // Bring impl<int, IPrint> and impl<double, IPrint> into scope. This means
  // we can use unqualified member lookup to find the print and to_string 
  // interface methods.
  using impl int, double : IPrint;

  // Call interface methods on arithmetic types! That's new.
  int x = 100;
  x.print();

  double y = 200.2;
  std::cout<< y.to_string()<< "\n";

  // Dynamic type erasure! That's new.
  dyn<IPrint>* p1 = make_dyn<IPrint>(new int { 300 });
  dyn<IPrint>* p2 = make_dyn<IPrint>(new double { 400.4 });
  p1->print();
  std::cout<< p2->to_string()<< "\n";
  delete p1;
  delete p2;

  // Tuple expressions are new.
  auto tup = (1, "Two", 3.3);

  // Tuple types are new.
  using Type = (int, const char*, double);
  static_assert(Type == decltype(tup));
}