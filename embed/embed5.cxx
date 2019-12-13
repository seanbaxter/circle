#include <cstdio>

// Load the file from a string non-type template parameter.
template<const char filename[]>
void do_shader() {
  static const char text[] = @embed(char, filename);
  printf("do_shader<%s> -> %zu bytes\n", filename, sizeof(text));
}

int main() {
  do_shader<"embed1.cxx">();
  do_shader<"embed2.cxx">();
  do_shader<"embed3.cxx">();
  do_shader<"embed4.cxx">();
  do_shader<"embed5.cxx">();
  return 0;
}