#include <string>
#include <algorithm>
#include <cstdio>

int main() {
  std::string s = "Hello world";
  printf("%c ", s[:])...;
  printf("\n");         // Prints 'H e l l o   w o r l d '

  // Loop over the first half of s in forward order and the back half of 
  // s in reverse order, swapping each pair.
  size_t mid = s.size() / 2;
  std::swap(s[:mid:1], s[:mid:-1])...;
  puts(s.c_str());      // Prints 'dlrow olleH'

  // Reverse it a second time. 
  // The :1 forward step is implicit so is dropped. The end iterator on 
  // the back half is dropped, because the expansion expression's loop count
  // is inferred from the shortest slice length. 
  std::swap(s[:mid], s[::-1])...;
  puts(s.c_str());      // Prints 'Hello world'

  // Reverse the string using list comprehension.
  std::string s2 = [s[::-1]...];
  puts(s2.c_str());     // Prints 'dlrow olleH'

  // Print the front half in forward order and the back half in reverse order.
  std::string s3 = [s[:mid]..., s[:mid:-1]...];
  puts(s3.c_str());     // Prints 'Hellodlrow '

  // Use list comprehension to collect the even index characters, then
  // the odd index characters. Uppercase the front half and lowercase the
  // back half.
  std::string s4 = [(char)toupper(s[::2])..., (char)tolower(s[1::2])...];
  puts(s4.c_str());     // Prints 'HLOWRDel ol'
}