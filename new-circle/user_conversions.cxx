#pragma feature as    // Enable as-expression.

struct S {
  // [no_implicit_user_conversion] only applies to non-explicit 
  // conversions.
  // explicit conversions must already be called explicitly or from a 
  // contextual conversion context.
  operator int() const noexcept;
  operator const char*() const noexcept;
  explicit operator bool() const noexcept;
};

void f1(int);
void f2(const char*);

int main() {
  S s;
  int x1 = s;
  const char* pc1 = s;

  #pragma feature no_implicit_user_conversions

  // Contextual conversions are permitted to use user-defined conversions.
  if(s) { }

  // Implicit user-defined conversions outside contextual conversions are
  // prohibited.
  int x2 = s;                         // Error
  const char* pc2 = s;                // Error
  f1(s);                              // Error
  f2(s);                              // Error
  
  // You may use as-expression to cast to a type.
  int x3 = s as int;                  // Ok
  const char* pc3 = s as const char*; // Ok
  f1(s as int);                       // Ok
  f2(s as const char*);               // Ok

  // You may use as-expression to permit implicit conversions.
  int x4 = s as _;                    // Ok
  const char* pc4 = s as _;           // Ok
  f1(s as _);                         // Ok
  f2(s as _);                         // Ok
}

