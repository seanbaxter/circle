#include <vector>
#include <string>
#include <cstdio>

int main() {
  std::string s1 = "Hello world";
  std::string s2 = "What's new?";

  // Emit pairs of elements.

  // Print 'H*e*l*l*o* *w*o*r*l*d*'.
  std::string s3 = [{s1[:], '*'}...];
  puts(s3.c_str());

  // Print 'H!e!l!l!o! !w!o!r!l!d!'.
  std::string s4 = [for c : s1 => {c, '!'}...];
  puts(s4.c_str());

  // Print each character with upper the lower case.
  // Print 'HhEeLlLlOo  WwOoRrLlDd'
  std::string s5 = [for c : s1 => { (char)toupper(c), (char)tolower(c) }...];
  puts(s5.c_str());

  // Intersperse elements from s1 with a constant.
  // Print 'H-e-l-l-o- -w-o-r-l-d'
  std::string s6 = [{s1[:-2], '-'}..., s1.back()];
  puts(s6.c_str());

  // Print 'H+e+l+l+o+ +w+o+r+l+d'
  std::string s7 = [for c : s1[:-2]... => { c, '+'}..., s1.back()];
  puts(s7.c_str());

  // Interleave two strings.
  // Print 'HWehlalto' sw onrelwd?'
  std::string s8 = [{s1[:], s2[:]}...];
  puts(s8.c_str());
}
