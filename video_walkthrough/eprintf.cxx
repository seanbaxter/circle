#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdio>

// Scan through until the closing '}'.
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

  throw std::runtime_error("mismatched { } in parse_braces");
}

// Edit the format specifier. Dump the text inside 
inline void transform_format(const char* fmt, std::string& fmt2, 
  std::vector<std::string>& exprs) {

  std::vector<char> text;
  while(char c = *fmt) {
    if('{' == c) {
      // Parse the contents of the braces.
      const char* end = parse_braces(fmt + 1);
      exprs.push_back(std::string(fmt + 1, end - 1));
      fmt = end;
      text.push_back('%');
      text.push_back('s');

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

@macro auto eprintf(const char* fmt) {
  // Process the input specifier. Remove {name} and replace with %s.
  // Store the names in the array.
  @meta std::vector<std::string> exprs;
  @meta std::string fmt2;
  @meta transform_format(fmt, fmt2, exprs);

  // Convert each name to an expression and from that to a string.
  // Pass to sprintf via format.
  return printf(
    @string(fmt2.c_str()), 
    std::to_string(@expression(@pack_nontype(exprs))).c_str()...
  );
}


int main() {
  double x = 5;
  eprintf("x = {x} sqrt = {sqrt(x)} exp = {exp(x)}\n");

  return 0;
}