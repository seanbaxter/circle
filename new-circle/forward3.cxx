// A function with a forwarding reference parameter.
void func(auto&& x) { }    // #1

#pragma feature forward

// A function with an rvalue reference parameter. This is a different
// overload from #1.
void func(auto&& x) { }    // #2

int main() {
  // Pass an lvalue. 
  // OK: This matches #1 and not #2.
  int x = 1;
  func(x);

  // Pass an xvalue.
  // ERROR: This is ambiguous, because it matches both #1 and #2.
  func(5);
}