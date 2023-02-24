#feature on adl

namespace ns {
  struct obj_t { };
  void func(obj_t obj);   // #1
}

int main() {
  // Ok. Unqualified lookup fails to find any declarations. ADL is not 
  // disqualified, so will be attempted at the point of the call.
  auto lift1 = [] func;

  // Error. [adl]: adl candidate called without adl token before unqualified name
  // The best viable candidate was found by ADL, but the adl prefix wasn't 
  // used.
  lift1(ns::obj_t());
  
  // A declaration to disqualify ADL, from [basic.lookup.argdep]/1.
  using func = int;       // #2

  // Error: func finds #2, which is not a function, so disqualifies 
  // ADL. The lifting lambda cannot be created, because its set of
  // candidates is empty.
  auto lift2 = [] func;

  // Ok. func finds #2, which is not a function. But because the adl prefix
  // is used, that declaration is discarded. The lifting lambda is created
  // ADL-capable. The actual lookup is deferred until its used.
  auto lift3 = [] adl func;

  // lift2 performs ADL at the point of the call. This is allowed, because
  // we used the adl token at its creation.
  lift3(ns::obj_t());
}