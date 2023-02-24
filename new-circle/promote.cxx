int main() {
  char           x = 1;
  unsigned char  y = 2;
  short          z = 3;
  unsigned short w = 4;

  // Integral types smaller than int are automatically promoted to int
  // before arithmetic.
  static_assert(int == decltype(x * x), "promote to int");
  static_assert(int == decltype(y * y), "promote to int");
  static_assert(int == decltype(z * z), "promote to int");
  static_assert(int == decltype(w * w), "promote to int");

  char8_t  a = 'a';
  char16_t b = 'b';
  char32_t c = 'c';
  wchar_t  d = 'd';
  static_assert(int      == decltype(a * a), "promote to int");
  static_assert(int      == decltype(b * b), "promote to int");
  static_assert(unsigned == decltype(c * c), "promote to unsigned");
  static_assert(int      == decltype(d * d), "promote to int");

  // Turn this very surprising behavior off.
  #feature on no_integral_promotions
  static_assert(char           == decltype(x * x), "does not promote to int");
  static_assert(unsigned char  == decltype(y * y), "does not promote to int");
  static_assert(short          == decltype(z * z), "does not promote to int");
  static_assert(unsigned short == decltype(w * w), "does not promote to int");

  static_assert(char8_t  == decltype(a * a), "does not promote to int");
  static_assert(char16_t == decltype(b * b), "does not promote to int");
  static_assert(char32_t == decltype(c * c), "does not promote to unsigned");
  static_assert(wchar_t  == decltype(d * d), "does not promote to int");
}
