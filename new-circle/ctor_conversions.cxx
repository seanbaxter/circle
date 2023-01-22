struct S {
  // [no_implicit_ctor_conversions] only applies to non-explicit
  // constructors.
  S(int i);
};

void func(S);

int main() {
  #pragma feature as no_implicit_ctor_conversions

  // Applies to implicit conversion sequences.
  func(1);       // Error
  func(1 as S);  // Ok
  func(1 as _);  // Ok  

  // Also applies to copy-initalization.
  S s1 = 1;      // Error
  S s2 = 1 as S; // Ok
  S s3 = 1 as _; // Ok
  S s4 = S(1);   // Ok
  S s5 { 1 };    // Ok
}