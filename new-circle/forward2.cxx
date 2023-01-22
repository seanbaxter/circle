// T&& is now freed up to mean rvalue reference.
#pragma feature forward

void f1(forward auto x);  // This is a forwarding parameter.
void f2(auto&& x);        // This is an rvalue reference parameter.

int main() {
  int x = 1;

  f1(1);   // Pass an xvalue to the forward parameter.
  f1(x);   // Pass an lvalue to the forward parameter.

  f2(1);   // Pass an xvalue to rvalue reference parameter.
  f2(x);   // Error: cannot pass an lvalue to the rvalue reference parameter.
}
