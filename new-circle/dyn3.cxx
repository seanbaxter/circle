#pragma feature interface forward self template_brackets
#include <memory>
#include <iostream>

// This is all goes into a library.

// Create a unique_ptr that wraps a dyn.
template<typename Type, interface IFace>
std::unique_ptr!dyn!IFace make_unique_dyn(forward auto... args) {
  return std::unique_ptr!dyn!IFace(make_dyn<IFace>(new Type(forward args...)));
}

// Implicitly generate a clone interface for copy-constructible types.
template<interface IFace>
interface IClone auto : IFace {
  // The default-clause causes SFINAE failure to protect the program from
  // being ill-foremd if IClone is attempted to be implicitly instantiated 
  // for a non-copy-constructible type.
  std::unique_ptr!dyn!IClone clone() const 
  default(Self~is_copy_constructible) {
    // Pass the const Self lvalue to make_unique_dyn, which causes copy
    // construction.
    return make_unique_dyn!<Self, IClone>(self);
  }
};

template<interface IFace>
class Box {
public:
  using Ptr = std::unique_ptr!dyn!IClone!IFace;

  Box() = default;

  // Allow direct initialization from unique_ptr!dyn!IClone!IFace.
  explicit Box(Ptr p) : p(std::move(p)) { }
  
  Box(Box&&) = default;
  Box(const Box& rhs) {
    // Copy constructor. This is how we clone.
    p = rhs.p->clone();
  }

  Box& operator=(Box&& rhs) = default;
  Box& operator=(const Box& rhs) {
    // Clone here too. We can't call the type erased type's assignment, 
    // because the lhs and rhs may have unrelated types that are only 
    // common in their implementation of IFace.
    p = rhs.p->clone();
    return self;
  }

  // Return a dyn<IFace>*. This is reached via upcast from dyn<IClone<IFace>>*.
  // It's possible because IFace is a base interface of IClone<IFace>.
  // If the user wants to clone the object, it should do so through the Box.
  dyn!IFace* operator->() noexcept {
    return p.get();
  }

  void reset() {
    p.reset();
  }

private:
  Ptr p;
};

template<typename Type, interface IFace>
Box!IFace make_box(forward auto... args) {
  return Box!IFace(make_unique_dyn!<Type, IClone!IFace>(forward args...));
}

// This is the user-written part. Very little boilerplate.
interface IText {
  void print() const;
  void set(std::string s);
  void to_uppercase();
};

impl std::string : IText {
  void print() const {
    // Print the address of the string and its contents.
    std::cout<< "string.IText::print ("<< &self<< ") = "<< self<< "\n";
  }
  void set(std::string s) {
    std::cout<< "string.IText::set called\n";
    self = std::move(s);
  }
  void to_uppercase() {
    std::cout<< "string.IText::to_uppercast called\n";
    for(char& c : self)
      c = std::toupper(c);
  }
};

int main() {
  Box x = make_box!<std::string, IText>("Hello dyn");
  x->print();

  // Copy construct a clone of x into y.
  Box y = x;

  // Mutate x.
  x->to_uppercase();

  // Print both x and y. y still refers to the original text.
  x->print();
  y->print();

  // Copy-assign y back into x, which has the original text.
  x = y;

  // Set a new text for y.
  y->set("A new text for y");

  // Print both.
  x->print();
  y->print();
}