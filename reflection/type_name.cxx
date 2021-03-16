#include <type_traits>
#include <cstdio>
#include <string>

int main() {
  int array[4];

  // Turn a type into a type string.
  printf("@type_string = %s\n", @type_string(decltype(array)));

  // Turn a type and string into a decl string.
  printf("@decl_string = %s\n", @decl_string(decltype(array), "array"));

  // Turn a string into a type.
  @type_id("int[5]") array2;
  printf("decltype(array2) = %s\n", @type_string(decltype(array2)));

  // Construct a meta string.
  @meta std::string s = "double";
  static_assert(std::is_same_v<double, @type_id(s)>);

  // Append function parameters to the string.
  @meta s += "(int, const char*)";
  static_assert(std::is_same_v<double(int, const char*), @type_id(s)>);
}