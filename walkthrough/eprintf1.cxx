#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <vector>
#include <iostream>

inline std::string vformat(const char* pattern, va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);

  int len = std::vsnprintf(nullptr, 0, pattern, args);
  std::string result(len, ' ');
  std::vsnprintf(result.data(), len + 1, pattern, args_copy);
 
  va_end(args_copy);
  return result;
}

inline std::string format(const char* pattern, ...) {
  va_list args;
  va_start(args, pattern);
  std::string s = vformat(pattern, args);
  va_end(args);
  return s;
}

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

@macro auto esprintf(const char* fmt) {
  // Process the input specifier. Remove {name} and replace with %s.
  // Store the names in the array.
  @meta std::vector<std::string> names;
  @meta std::string fmt2;
  @meta transform_format(fmt, fmt2, names);

  // Convert each name to an expression and from that to a string.
  // Pass to sprintf via format.
  return format(
    @string(fmt2.c_str()), 
    std::to_string(@expression(@pack_nontype(names))).c_str()...
  );
}

@macro auto eprintf(const char* fmt) {
  return std::cout<< esprintf(fmt);
}

@macro auto operator ""_e(const char* fmt, size_t len) {
  return esprintf(fmt);
}

int main() {
  double x = 5;
  std::cout<< "x = {x} sqrt = {sqrt(x)} exp = {exp(x)}\n"_e;

  return 0;
}