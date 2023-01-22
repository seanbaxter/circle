#include <iostream>

namespace ns1 {
  struct item_t { };

  // This is found with ADL.
  void f(item_t) {
    std::cout<< "called ns1::f(item_t)\n";
  }
}

namespace ns2 {
  void f(double) {
    std::cout<< "called ns2::f(double)\n";
  }
};

template<typename T>
void f(T) {
  std::cout<< "called ::f({})\n".format(T~string);
}

void doit(auto callable, auto arg) {
  // Invoke the lifting lambda. 
  // * If the lambda was formed with unqualified lookup, ADL is used to
  //   find the candidate. Using-declarations encountered during unqualified
  //   lookup may inject additional candidates for overload resolution.
  // * If the lamdba was formed with qualified lookup, ADL is not used.
  callable(arg);
}

int main() {
  // Make an ADL call to f. The argument type int has no associated 
  // namespaces, so only ::f is a candidate.
  doit([]f, 1);

  // Make an ADL call to f. The argument type has ns1 as an associated
  // namespace. Both ::f and ns1::f are candidates, but ns1::f is the
  // better match.
  doit([]f, ns1::item_t{});

  // Make a qualified call to f. The associated namespaces of item_t aren't
  // considered, because ADL only happens with unqualified lookup.
  doit([]::f, ns1::item_t{});

  // Unqualified name lookup finds the alias-declaration for ns2::f.
  // This becomes one of the candidates, even though it's not a member of 
  // an associated of the argument type double. This is exactly the 
  // std::swap trick.
  using ns2::f;
  doit([]f, 3.14);
}

