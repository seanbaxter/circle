#include <string>
#include <vector>
#include <cstdio>

int main() {
  std::string s = "Hello world";

  // Use a for-expression to print only the lowercase characters.
  std::string s2 = ['*', for c : s if islower(c) => c..., '*'];
  puts(s2.c_str());     // Prints '*elloworld*'

  // Use a for-expression to double each character.
  std::string s3 = [for c : s => { c, c }...];
  puts(s3.c_str());     // Prints 'HHeelllloo  wwoorrlldd'

  // Use a for-expression to emit upper/lower-case pairs.
  std::string s4 = [for c : s => {(char)toupper(c), (char)tolower(c)}...];
  puts(s4.c_str());     // Prints 'HhEeLlLlOo  WwOoRrLlDd'

  // Use the index to create alternating upper and lowercase characters.
  std::string s5 = [for i, c : s => (char)((1&i) ? tolower(c) : toupper(c))...];
  puts(s5.c_str());     // Prints 'HeLlO WoRlD'

  // Create a vector of vectors.
  printf("Creating a vector of vectors:\n");
  std::vector vecs = [for i : 5 => [for i2 : i => i...] ...];
  for(auto& v : vecs) {
    printf("[ "); printf("%d ", v[:])...; printf("]\n");
  }
}