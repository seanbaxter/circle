#include <cstdio>

template<typename... types_t>
void func0() {
  // Loop over parameter packs using sizeof... and ...[].
  printf("func0:\n");
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    printf("  %s\n", types_t...[i].string);
}

void func1() {
  printf("func1:\n");
  @meta for typename(type_t : { char, double, long[3], char(short) })
    printf("  %s\n", type_t.string);
}

template<typename... types_t>
void func2() {
  // Loop over parameter pack using for-typename.
  printf("func2:\n");
  @meta for typename(type_t : { types_t... })
    printf("  %s\n", type_t.string);
}

template<typename... types_t>
void func3() {
  // Loop over an assortment of types inside braces.
  // Hit an int at the start, then the even elements of types_t, then
  // a char[1].
  printf("func3:\n");
  @meta for typename(type_t : { int, types_t...[::2] ..., char[1]})
    printf("  %s\n", type_t.string);
}

enum typename my_types_t {
  int, void, char, wchar_t[10]
};
template<typename list_t>
void func4() {
  static_assert(__is_typed_enum(list_t));

  // Expand a typed enum into your for typename braces.
  printf("func4:\n");
  @meta for typename(type_t : { long, @enum_types(list_t)..., char16_t })
    printf("  %s\n", type_t.string);
}

template<typename list_t>
void func5() {
  static_assert(__is_typed_enum(list_t));

  // Alternatively, ditch the braces and use the 'enum' keyword to 
  // specify we want iteration over all types in the typed enum.
  printf("func5:\n");
  @meta for typename(type_t : enum list_t)
    printf("  %s\n", type_t.string);
}

int main() {
  func0<double, int*, char[3], short(long)>();
  func1();
  func2<float, short*, void*()>();
  func3<char, unsigned char, signed char, char16_t, char32_t>();
  func4<my_types_t>();
  func5<my_types_t>();
}