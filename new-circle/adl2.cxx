#pragma feature adl

namespace ns {
  struct obj_t { 
    void f();

    void func(int);      // #1
  };

  void func(obj_t obj);  // #2
}

void ns::obj_t::f() {
  // Error: [adl]: no arguments in adl call have associated namespaces.
  adl func(1, "Hello world");

  // Tries to call #1
  // Error: cannot convert lvalue ns::obj_t to int.
  func(*this);

  // [basic.lookup.argdep] dictates that if unqualified 
  // lookup finds any of:
  // * declaration of a class member, or
  // * function declaration inhabiting a block scope, or
  // * declaration not of a function or function template
  // then ADL is not used.
  // But we used the adl keyword, meaning we really want ADL!
  // Therefore, when any of these three things is found by
  // unqualified lookup, the compiler discards the declaration 
  // and goes straight to ADL.

  // Calls #2 successfully.
  adl func(*this);
}