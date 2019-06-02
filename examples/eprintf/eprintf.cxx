#include "../include/format.hxx"
#include <cmath>

// Parse out the expression names and replace each name with %, which is the
// escape code for cirformat.

inline const char* parse_braces(const char* text) {
  const char* begin = text;

  while(char c = *text) {
    if('{' == c)
      return parse_braces(text + 1);
    else if('}' == c)
      return text + 1;
    else
      ++text;    
  }

  printf("mismatched { } in \"%s\"", text);
  abort();
}

inline void transform_format(const char* fmt, std::string& fmt2, 
  std::vector<std::string>& names) {

  std::vector<char> text;
  while(char c = *fmt) {
    if('{' == c) {
      // Parse the contents of the braces.
      const char* end = parse_braces(fmt + 1);
      names.push_back(std::string(fmt + 1, end - 1));
      fmt = end;
      text.push_back('%');

    } else if('%' == c && '{' == fmt[1]) {
      // %{ is the way to include a { character in the format string.
      fmt += 2;
      text.push_back('{');

    } else {
      ++fmt;
      text.push_back(c);
    }
  }

  fmt2 = std::string(text.begin(), text.end());
}

@macro auto eprintf(const char* __fmt) {
  // Use __ before all names in this macro. We don't want to accidentally
  // shadow names in the calling scope.
  @meta std::string __fmt2;
  @meta std::vector<std::string> __names;
  @meta transform_format(__fmt, __fmt2, __names);

  return cirprint(
    __fmt2.c_str(), 
    @expression(__names[__integer_pack(__names.size())])...
  );
}

int main() {
  int x = 101;
  double y = 3.14;

  eprintf("x = {x}, y = {y}, z = {sqrt(1000.0)}\n");

  return 0;
}