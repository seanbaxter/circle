#pragma feature as        // Permit any implicit conversion with "x as _".
void f_short(short x);
void f_int(int x);
void f_unsigned(unsigned x);
void f_long(long x);
void f_float(float x);
void f_double(double x);
void f_bool(bool x);

int main() {
  #pragma feature no_implicit_integral_narrowing
  int x_int = 100;
  f_short(x_int);              // Error
  f_short(x_int as _);         // OK
  #pragma feature_off no_implicit_integral_narrowing

  #pragma feature no_implicit_floating_narrowing
  double x_double = 100;
  f_float(x_double);           // Error
  f_float(x_double as _);      // Ok
  #pragma feature_off no_implicit_floating_narrowing

  #pragma feature no_implicit_signed_to_unsigned
  f_unsigned(x_int);           // Error
  f_unsigned(x_int as _);      // OK
  f_unsigned(x_double);        // Error
  f_unsigned(x_double as _);   // OK
  #pragma feature_off no_implicit_signed_to_unsigned

  #pragma feature no_implicit_widening
  char x_char = 'x';
  f_short(x_char);             // Error
  f_short(x_char as _);        // OK
  f_long(x_int);               // Error
  f_long(x_int as _);          // OK
  float x_float = 1;  
  f_double(x_float);           // Error
  f_double(x_float as _);      // OK
  #pragma feature_off no_implicit_widening

  #pragma feature as no_implicit_enum_to_underlying
  enum numbers_t : int {
    Zero, One, Two, Three,
  };

  f_int(Zero);                 // Error
  f_int(Zero as _);            // OK

  f_int(numbers_t::Zero);      // Error
  f_int(numbers_t::Zero as _); // OK 
  #pragma feature_off no_implicit_enum_to_underlying 

  // Use as _ to allow implicit narrowing conversions inside 
  // braced-initializer.
  short s1 { x_int };           // Error
  short s2 { x_int as _ };      // OK
  f_short({ x_int });           // Error
  f_short({ x_int as _});       // OK
  #pragma feature_off no_implicit_enum_to_underlying

  // Implicit conversions from pointers to bools are permitted by C++.
  const char* p = nullptr;
  f_bool(p);                    // OK
  #pragma feature no_implicit_pointer_to_bool
  // They are disabled by [no_implicit_pointer_to_bool]
  f_bool(p);                    // Error
  f_bool(p as bool);            // OK
  f_bool(p as _);               // OK
};