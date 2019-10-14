#include <cstdio>
#include <string>
#include <algorithm>

template<const char name[]>
void func() {
  @meta printf("func<%s> instantiated\n", name);
}

int main() {
  // Instantiate on a string literal.
  func<"My string literal">();

  // Instantiate on a constexpr or meta const char array.
  constexpr char array[] = "My constexpr string";
  func<array>();

  @meta char array2[] = "My meta string";
  func<array2>();

  // Instantiate on a constexpr or meta pointer to character.
  constexpr const char* ptr = "My constexpr string";
  func<ptr>();

  @meta const char* ptr2 = "My meta string";
  func<ptr2>();

  // Instantiate on a meta std::string. This is not a literal type so must
  // be made meta.
  @meta std::string s = "My meta string";
  func<s>();

  // The strings can be modified at compile time. They can come from any
  // source. They don't have to refer to literals!
  @meta std::transform(s.begin(), s.end(), s.begin(), toupper); 
  func<s>();
  
  return 0;
}
