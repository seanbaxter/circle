#pragma once
#include "../include/serialize.hxx"
#include <sstream>
#include <cassert>

// Scan ahead to the next escape character '%'
inline const char* scan_to_escape(const char* p) {
  while(*p && '%' != *p)
    ++p;
  return p;
}

template<typename... args_t>
@meta std::string cirformat(@meta const char* fmt, args_t&&... args) {
  const size_t num_args = sizeof...(args);
  @meta size_t len = strlen(fmt);
  @meta size_t cur_arg = 0;

  // The output stream is runtime.
  std::ostringstream oss;

  @meta+ while(*fmt) {
    // The @meta+ keyword makes this entire block recursively @meta, unless
    // opted-out by @emit.
    if('%' == fmt[0] && '%' != fmt[1]) {

      // Check that we aren't doing an out-of-bounds pack index.
      static_assert(cur_arg < num_args, 
        "cirformat replacement is out-of-range");

      // Stream the argument to the output. Use indent=-1 to turn off newlines.
      @emit stream_simple(oss, args...[cur_arg]);

      // Advance to the next argument and format specifier character.
      ++cur_arg;
      ++fmt;

    } else {
      // In an %% sequence, move past the first %.
      if('%' == fmt[0])
        ++fmt;

      // Scan to the next format specifier or the end of the format string.
      const char* end = scan_to_escape(fmt + 1);

      // Stream a string literal of the substring between the escapes.
      @emit oss<< @string(std::string(fmt, end - fmt));

      // Advance the format string to the end of this substring.
      fmt = end;
    }
  }

  static_assert(cur_arg == num_args, 
    "not all cirformat arguments used in format string");

  return oss.str();
}

template<typename... args_t>
@meta int fcirprint(FILE* stream, @meta const char* fmt, args_t&&... args) {
  std::string s = cirformat(fmt, std::forward<args_t>(args)...);
  fputs(s.c_str(), stream);
  return s.size();
}

template<typename... args_t>
@meta int cirprint(@meta const char* fmt, args_t&&... args) {
  return fcirprint(stdout, fmt, std::forward<args_t>(args)...);
}
