#pragma once
#include <vector>
#include <cstdarg>
#include <cstring>
#include <array>
#include <list>
#include <map>
#include <set>
#include <optional>
#include <string>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <stdexcept>

// https://fmt.dev/latest/syntax.html

// format_spec ::=  [[fill]align][sign]["#"]["0"][width]["." precision][type]
// fill        ::=  <a character other than '{', '}' or '\0'>
// align       ::=  "<" | ">" | "=" | "^"
// sign        ::=  "+" | "-" | " "
// width       ::=  integer | "{" expression "}"
// precision   ::=  integer | "{" expression "}"
// type        ::=  int_type | "a" | "A" | "c" | "e" | "E" | "f" | "F" | "g" | "G" | "p" | "s"
// int_type    ::=  "b" | "B" | "d" | "n" | "o" | "x" | "X"

namespace cirfmt {

struct range_t {
  const char* begin;
  const char* end;

  explicit operator bool() const {
    return begin < end;
  }

  size_t size() { return end - begin; }

  bool peek_if(char c) {
    return begin < end && begin[0] == c;
  }
  bool advance_if(char c) {
    bool success = peek_if(c);
    if(success) ++begin;
    return success;
  }
  bool advance_if(const char* s) {
    size_t len = strlen(s);
    bool status = size() >= len && 0 == memcmp(begin, s, len);
    if(status)
      begin += len;
    return status;
  }

  char next() {
    return begin < end ? *begin++ : 0;
  }

  char operator[](ptrdiff_t index) {
    return begin + index < end ? begin[index] : 0;
  }

  template<typename type_t>
  void advance(const type_t& obj) {
    begin = obj->range.end;
  }
  void advance(const char* p) {
    begin = p;
  }
};

template<typename attr_t>
struct result_base_t {
  range_t range;
  attr_t attr;

  result_base_t() { }
};

template<typename attr_t>
struct result_t : protected result_base_t<attr_t> {
  result_t() : success(false) { }
  result_t(range_t range, attr_t attr) : success(true) { 
    this->range = range;
    this->attr = std::move(attr);
  }

  explicit operator bool() const { 
    return success;
  }

  // Access the attr and next members through this operator. This checks
  // success to confirm we're using the result correctly.
  result_base_t<attr_t>* operator->() {
    assert(success);
    return this;
  }
  const result_base_t<attr_t>* operator->() const {
    assert(success);
    return this;
  }
  attr_t& operator*() { 
    assert(success);
    return this->attr;
  }

private:
  bool success;
};

template<typename attr_t>
result_t<attr_t> make_result(range_t range, attr_t attr) {
  return result_t(range, std::move(attr));
}
template<typename attr_t>
result_t<attr_t> make_result(const char* begin, const char* end, attr_t attr) {
  return result_t(range_t { begin, end }, std::move(attr));
}

////////////////////////////////////////////////////////////////////////////////
// Retrieve the name of an enumerator from the enum.

template<typename type_t>
const char* enum_to_name(type_t e) {
  switch(e) {
    @meta for enum(type_t e2 : type_t) {
      case e2:
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////

struct fmt_t {
  // In all cases, 0 is the default value.
  char fill;          // any character other than {, } or \0.
  char align;         // <, >, = or ^.
  char sign;          // +, - or <space>
  bool alt_form;      // #
  bool zero_padding;  // 0
  char type;
  int width;
  int precision = 5;  // Use 5 decimal places for default precision.
};

// va_arg-based format for error messages.
inline std::string va_format(const char* pattern, ...) {
  va_list args;
  va_start(args, pattern);

  va_list args_copy;
  va_copy(args_copy, args);

  int len = std::vsnprintf(nullptr, 0, pattern, args);
  std::string result(len, ' ');
  std::vsnprintf((char*)result.data(), len + 1, pattern, args_copy);

  va_end(args_copy);
  va_end(args);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Print decimal and hexadecimal integers.

template<typename type_t>
int print_bin(type_t x, char* text) {
  static_assert(std::is_integral_v<type_t>);

  bool is_signed = x < 0;
  if(is_signed) {
    *text++ = '-';
    x = -x;
  }

  int count = 0;
  do {
    text[count++] = '0' + (x % 2);
    x /= 2;
  } while(x);

  std::reverse(text, text + count);
  return count + (int)is_signed;
}

template<typename type_t>
int print_oct(type_t x, char* text) {
  static_assert(std::is_integral_v<type_t>);

  bool is_signed = x < 0;
  if(is_signed) {
    *text++ = '-';
    x = -x;
  }

  int count = 0;
  do {
    text[count++] = '0' + (x % 8);
    x /= 8;
  } while(x);

  std::reverse(text, text + count);
  return count + (int)is_signed;
}

template<typename type_t>
int print_dec(type_t x, char* text) {
  static_assert(std::is_integral_v<type_t>);

  bool is_signed = x < 0;
  if(is_signed) {
    *text++ = '-';
    x = -x;
  }

  int count = 0;
  do {
    text[count++] = '0' + (x % 10);
    x /= 10;
  } while(x);

  std::reverse(text, text + count);
  return count + (int)is_signed;
}

template<typename type_t>
int print_hex(type_t x, bool upper, char* text) {
  static_assert(std::is_integral_v<type_t>);

  bool is_signed = x < 0;
  if(is_signed) {
    *text++ = '-';
    x = -x;
  }

  int count = 0;
  do {
    int digit = x % 16;
    char c = digit < 10 ? 
      '0' + digit :
      (upper ? 'A' : 'a') + digit - 10;

    text[count++] = c;
    x /= 16;
  } while(x);

  std::reverse(text, text + count);
  return count + (int)is_signed;
}

////////////////////////////////////////////////////////////////////////////////
// Print decimal and hexadecimal floats.

// Use frexp and logarithm conversions to put the float in base-10 and 
// base-16 scientific notation. (i.e. a single leading digit).
inline double exp16(double x) {
  return exp2(4 * x);
}

inline double log16(double x) {
  return log2(x) / 4;
}

inline double frexp10(double x, int* exp) {
  double y = frexp(x, exp);
  if(y) {
    double y2 = exp2(*exp);

    *exp = (int)floor(log10(y * y2));
    y *= exp10(-*exp) * y2; 
  }
  return y;
}

inline double frexp16(double x, int* exp) {
  double y = frexp(x, exp);
  if(y) {
    double y2 = exp2(*exp);

    *exp = (int)floor(log16(y * y2));
    y *= exp16(-*exp) * y2; 
  }
  return y;
}


// Print the whole number part of fixed float.
inline int print_dec_integral_part(double x, char* text) {
  assert(x >= 0);

  int count = 0;
  while(!count || x >= 1) {
    text[count++] = '0' + (int)fmod(x, 10);
    x /= 10;
  }
  std::reverse(text, text + count);
  return count;
}

// Print the input in exactly count characters. This is for printing the
// fractional part of a float, or for printing all digits in a binary value.
inline void print_dec(unsigned long x, int count, char* text) {
  for(int i = 0; i < count; ++i) {
    text[count - i - 1] = '0' + (x % 10);
    x /= 10;
  }
}

inline void print_hex(unsigned long x, int count, bool upper, char* text) {
  for(int i = 0; i < count; ++i) {
    int digit = x % 16;
    char c = digit < 10 ? 
      '0' + digit :
      (upper ? 'A' : 'a') + digit - 10;

    text[count - i - 1] = c;
    x /= 16;
  }
}

// Print no more than prec digits.
inline int print_dec_exp(double x, int prec, bool upper, char* text) {
  assert(x >= 0);

  // Print the integral part.
  int exp;
  double y = frexp10(x, &exp);

  int digit = (int)y;
  int count = 0;
  text[count++] = '0' + digit;
  text[count++] = '.';

  // Print the fractional part.
  y *= 10;
  unsigned long frac = (unsigned long)(pow(10, prec) * y);
  print_dec(frac, prec, text + count);
  count += prec;

  // Print the exponent.
  text[count++] = upper ? 'E' : 'e';
  text[count++] = exp >= 0 ? '+' : '-';
  count += print_dec((unsigned long)abs(exp), text + count);

  return count;  
}

inline int print_hex_exp(double x, int prec, bool upper, char* text) {
  assert(x >= 0);

  // Print the integral part.
  int exp;
  double y = frexp16(x, &exp);

  int digit = (int)y;

  int count = 0;
  text[count++] = '0';
  text[count++] = upper ? 'X' : 'x';
  text[count++] = digit < 10 ? 
    '0' + digit :
    (upper ? 'A' : 'a') + digit - 10;
  text[count++] = '.';

  // Print the fractional part.
  y *= 16;
  unsigned long frac = (unsigned long)(pow(16, prec) * y);
  print_hex(frac, prec, upper, text + count);
  count += prec;

  // Print the exponent.
  text[count++] = upper ? 'P' : 'p';
  text[count++] = exp >= 0 ? '+' : '-';
  count += print_hex((unsigned long)abs(exp), upper, text + count);

  return count;  
}

int print_floating(fmt_t fmt, double x, char* text) {
  int count = 0;
  switch(fmt.type) {
    case 'F':
    case 'f': {
      // Decimal floating-point value.
      double frac, integral;
      frac = std::modf(x, &integral);

      // Print the integral part.
      count = print_dec_integral_part(integral, text);

      // Print the decimal.
      text[count++] = '.';

      // Set 16 as the max precision. This will exhaust all 52 mantissa bits.
      if(fmt.precision > 16)
        fmt.precision = 16;

      // Print the fractional part.
      unsigned long frac2 = (unsigned long)(pow(10, fmt.precision) * frac);
      print_dec(frac2, fmt.precision, text + count);
      count += fmt.precision;
      break;
    }

    case 'E':
    case 'e': {
      // Set 16 as the max precision. This will exhaust all 52 mantissa bits.
      if(fmt.precision > 16)
        fmt.precision = 16;

      // Print in scientific notation.
      count = print_dec_exp(x, fmt.precision, isupper(fmt.type), text);
      break;
    }

    case 'A':
    case 'a': {
      // Set 13 as the max precision. This will exhaust all 52 mantissa bits.
      if(fmt.precision > 16)
        fmt.precision = 16;

      // Print in hexadecimal scientific notation.
      count = print_hex_exp(x, fmt.precision, isupper(fmt.type), text);
      break;
    }

    case 'G':
    case 'g': {
      // General format. Choose either 'f' or 'e'. 
      int exp;
      frexp10(x, &exp);
      char fmt2 = exp <= 5 && exp >= -1 ? 'f' : 'e';
      fmt.type = isupper(fmt.type) ? toupper(fmt2) : fmt2;
      count = print_floating(fmt, x, text);
      break;
    }

    case '%': {
      // Same as f, but multiply by 100 and print with %.
      fmt.type = 'f';
      count = print_floating(fmt, 100 * x, text);
      text[count++] = '%';
      break;
    }
  }

  return count;
}

template<typename type_t>
int print_integer(fmt_t fmt, type_t x, char* text) {
  int count = 0;
  switch(fmt.type) {
    case 'B':
    case 'b': {
      // Binary output.
      count = print_bin(std::make_unsigned_t<type_t>(x), text);
      break;
    }

    case 'O':
    case 'o': {
      count = print_oct(std::make_unsigned_t<type_t>(x), text);
      break;
    }

    case 'X':
    case 'x': {
      count = print_hex(std::make_unsigned_t<type_t>(x), isupper(fmt.type),
      text);
      break;
    }

    case 'd':
    case 'n': {
      count = print_dec(x, text);
      break;
    }
  }

  return count;
}

////////////////////////////////////////////////////////////////////////////////

enum format_type_t {
  format_type_default,
  format_type_int,
  format_type_float,
  format_type_pointer,
};

// width is minimum string length
// for non-number types, precision is maximum string length.

struct format_arg_t {
  // Offset within the modified format specifier at which to insert this
  // argument.
  format_type_t type;
  int index;
  fmt_t fmt;

  // Text of the expression.
  std::string width_expr;
  std::string precision_expr;

  bool pack;
  std::string expr;
};

// A non-pack argument.
template<format_type_t fmt_type_, int index_, typename type_t>
struct arg_t {
  static constexpr format_type_t fmt_type = fmt_type_;
  static constexpr int index = index_;

  fmt_t fmt;
  const type_t& obj;
};

template<format_type_t fmt_type, int index, typename type_t>
arg_t<fmt_type, index, type_t> make_arg(fmt_t fmt, const type_t& obj) {
  return { fmt, obj };
}

template<format_type_t fmt_type, int index, typename type_t>
arg_t<fmt_type, index, type_t> make_arg(fmt_t fmt, int width, int precision, 
  const type_t& obj) {

  fmt.width = width;
  fmt.precision = precision;

  return { fmt, obj };
}

// A pack argument.
template<format_type_t fmt_type_, int index_, typename... types_t>
struct pack_t {
  static constexpr format_type_t fmt_type = fmt_type_;
  static constexpr int index = index_;

  fmt_t fmt;
  const types_t& @(int...) ...;
};

template<format_type_t fmt_type, int index, typename... types_t>
pack_t<fmt_type, index, types_t...> make_pack(fmt_t fmt, int width, 
  int precision, const types_t&... objs) {

  fmt.width = width;
  fmt.precision = precision;

  return { fmt, objs... };
}

// Format builder.
struct fmt_builder_t {

  result_t<range_t> parse_braces(range_t range);  
  const char* advance_brace(range_t range);

  template<typename type_t>
  result_t<type_t> parse_integer(range_t range);

  result_t<format_arg_t> parse_arg(range_t range);

  void parse();

  void throw_error(const char* pos, const char* error);
  void throw_error(int offset, const char* error);

  // The original format specifier text.
  const char* format;

  // The rebuilt format specifier text. This gets streamed to the output, 
  // interrupted by escaped arguments.
  std::string format2;

  std::vector<format_arg_t> args;

};

inline result_t<range_t> fmt_builder_t::parse_braces(range_t range) {
  const char* begin = range.begin;
  result_t<range_t> result;

  if(range.advance_if('{')) {
    range.advance(advance_brace(range));
    result = make_result(begin, range.begin, 
      range_t { begin + 1, range.begin - 1});
  }

  return result;
}

inline const char* fmt_builder_t::advance_brace(range_t range) {
  int brace_count = 1;    
  while(range && brace_count) {
    switch(char c = range.next()) {
      case '{': ++brace_count; break;
      case '}': --brace_count; break;
      default:                 break;
    }
  }
  if(brace_count > 0)
    throw_error(range.begin, "unmatched open brace in format specifier"); 

  return range.begin;
}

template<typename type_t>
inline result_t<type_t> fmt_builder_t::parse_integer(range_t range) {
  type_t x = 0;
  const char* begin = range.begin;

  while(char c = range[0]) {
    if(c >= '0' && c <= '9') {
      type_t x2 = 10 * x + c - '0';

      // Fail if the size_t overflows.
      if(x2 < x) {
        throw_error(begin, "integer overflow in format specifier");
        break;
      }

      x = x2;
      ++range.begin;

    } else
      break;
  }

  result_t<type_t> result;
  if(range.begin > begin)
    result = make_result(begin, range.begin, x);

  return result;
}

inline result_t<format_arg_t> fmt_builder_t::parse_arg(range_t range) {
  const char* begin = range.begin;
  format_arg_t arg { };

  // The fill character can be any character other than ‘{‘, ‘}’ or ‘\0’. 
  // The presence of a fill character is signaled by the character following it,
  // which must be one of the alignment options. If the second character of 
  // format_spec is not a valid alignment option, then it is assumed that both
  // the fill character and the alignment option are absent.
  switch(char align = range[1]) {
    // Match an align with a preceding fill character.
    case '<':
    case '>':
    case '^':
      arg.fmt.fill = range.begin[0];
      arg.fmt.align = align;
      range.begin += 2;
      break;

    default:
      switch(char align = range[0]) {
        // Match an align with no preceding fill character.
        case '<':
        case '>':
        case '^':
          arg.fmt.align = align;
          range.begin += 1;
          break;
      }
      break;
  }

  // Parse the sign.
  switch(char sign = range[0]) {
    case '+':
    case '-':
    case ' ':
      arg.fmt.sign = sign;
      ++range.begin;
      break;

    default:
      break;
  }

  // Match the "alternate form" for numeric printing.
  arg.fmt.alt_form = range.advance_if('#');

  // Match the zero-padding.
  arg.fmt.zero_padding = range.advance_if('0');

  // Match the width argument.
  if(auto brace = parse_braces(range)) {
    range.advance(brace);
    if(!brace->attr)
      throw_error(brace->attr.begin, "expected width-specifier");

    arg.width_expr = { brace->attr.begin, brace->attr.end };

  } else if(auto x = parse_integer<int>(range)) {
    range.advance(x);
    arg.fmt.width = *x;
  }

  // Match the precision argument.
  if(range.advance_if('.')) {
    if(auto brace = parse_braces(range)) {
      range.advance(brace);
      if(!brace->attr)
        throw_error(brace->attr.begin, "expected precision-specifier");

      arg.precision_expr = { brace->attr.begin, brace->attr.end };

    } else if(auto x = parse_integer<int>(range)) {
      range.advance(x);
      arg.fmt.precision = *x;

    } else 
      throw_error(range.begin, "expected precision-specifier");
  }

  // Match a recognized type.
  switch(char type = range[0]) {
    // Integer formats:
    case 'b': case 'B': // Binary integer.
    case 'd':           // Decimal integer
    case 'o':           // Octal integer
    case 'x': case 'X': // Hexadecimal integer
    case 'n':           // Decimal with locale settings for separators.
      arg.fmt.type = type;
      arg.type = format_type_int;
      ++range.begin;
      break;

    // Floating-point formats:
    case 'a': case 'A': // Hexadecimal floating point format.
    case 'e': case 'E': // Exponent notation.
    case 'f': case 'F': // Fixed point.
    case 'g': case 'G': // General format.
    case '%':           // Fixed point as a percentage. Multiply by 100 and print %.
      arg.fmt.type = type;
      arg.type = format_type_float;
      ++range.begin;
      break;

    // Pointer:
    case 'p': case 'P':
      arg.fmt.type = type;
      arg.type = format_type_pointer;
      ++range.begin;
      break;

    case '}':
      // Allow '}' to close the format specifier.
      break;

    case 0:
      // Fail on the null terminator.
      throw_error(begin, "broken format specifier--close your braces");

    default: {
      // Any remaining character is an error.
      std::string msg = va_format("unrecognized type format code '%c'", type);
      throw_error(range.begin, msg.c_str());
    }
  }

  // Check for compatibility issues.
  // Binary and hex formatting not compatible with signs.
  if(arg.fmt.sign) {
    if(tolower(arg.type) == 'b')
      throw_error(begin, "binary value incompatible with sign attribute");

    if(tolower(arg.type) == 'o')
      throw_error(begin, "octal value incompatible with sign attribute");

    if(tolower(arg.type) == 'x')
      throw_error(begin, "hexadecimal value incompatible with sign attribute");
  }

  return make_result(begin, range.begin, std::move(arg));
}

inline void fmt_builder_t::parse() {
  const char* fmt = format;
  range_t range { fmt, fmt + strlen(fmt) };

  // Scan over all characters.
  while(char c = range.next()) {
    if('{' == c) {
      if(range.advance_if('{')) {
        // {{ is the escape for {.
        format2.push_back('{');

      } else {
        // Each argument starts with the expression. 
        // Use @parse_expression to get the 

        // Match a pack-expansion operator.
        bool is_pack = range.advance_if("...");

        // Parse the expression. @parse_expression returns the number of 
        // characters in an expression evaluation.
        size_t count = @@parse_expression(range.begin);
        std::string expr = std::string(range.begin, range.begin + count);
        range.begin += count;

        // Parse the format specifier.
        format_arg_t arg { };
        if(range.advance_if(':')) {
          // We have a format specifier.
          auto arg_ = parse_arg(range);
          range.advance(arg_);
          arg = std::move(arg_->attr);
        }

        if(!range)
          throw_error(range.begin, "unexpected end of format specifier");
        else if(!range.advance_if('}'))
          throw_error(range.begin, "expected '}' to close format specifier");

        arg.pack = is_pack;
        arg.expr = std::move(expr);

        // Insert this argument after these characters of the new format
        // specifier.
        arg.index = format2.size();

        // Push this argument to the builder.
        args.push_back(std::move(arg));
      }

    } else
      format2.push_back(c);
  }
}

inline void fmt_builder_t::throw_error(int offset, const char* error) {
  // Locate the cursor within newline characters of the format spec.
  const char* begin = format;

  int cur = 0;
  while(cur < offset) {
    if('\n' == format[cur++])
      begin = format + cur;
  }
  offset -= begin - format;

  // Locate the first newline to the right of offset.
  const char* end = begin;
  while(*end && '\n' != *end) ++end;

  // TODO: Compute caret adjustment to deal with tabs and UCS encodings.
  // One byte/column is not accurate.

  // Print the original error message plus the format specifier section and 
  // a caret.
  std::string msg = va_format("%s\n%.*s\n%*s^", error, end - begin, begin, 
    offset, "");
  throw std::runtime_error(std::move(msg));
}

inline void fmt_builder_t::throw_error(const char* pos, const char* error) {
  throw_error(pos - format, error);
}

////////////////////////////////////////////////////////////////////////////////

void stream_text(const fmt_t& fmt, char default_align, const char* begin, 
  int len, std::string& s) {

  int width = fmt.width;
  if(width > len) {
    // Insert fmt.width fill characters into s.
    char c = fmt.fill ? fmt.fill : ' ';
    int index = s.size();
    s.insert(s.end(), width, c);

    // Use fmt.align if one exists. Otherwise use default_align.
    int offset = index;
    switch(fmt.align ? fmt.align : default_align) {

      // Right-align the output.
      case '>':
        offset += width - len;
        break;

      // Center the output.
      case '^':
        offset += (width - len) / 2;
        break;

      // Left-align is the default.
      case '<':
      default:
        break;
    }
    memcpy(s.data() + offset, begin, len);
  
  } else {
    s.insert(s.end(), begin, begin + len);
  }
}

template<typename type_t>
void stream_integer(const fmt_t& fmt, const type_t& obj, std::string& s) {
  // The longest integer is 66 bits, for a 64-bit binary int with 0b header.
  char text[66];
  int count = print_integer(fmt, obj, text);
  stream_text(fmt, '>', text, count, s);
}

template<typename type_t>
void stream_floating(const fmt_t& fmt, const type_t& obj, std::string& s) {
  char text[64];
  int count = print_floating(fmt, obj, text);
  stream_text(fmt, '>', text, count, s);
}

template<typename type_t>
void stream_pointer(const fmt_t& fmt, const type_t& obj, std::string& s) {
  // Print all 16 hex digits of a 64-bit pointer. Prefix with 0x.
  char text[18];
  text[0] = '0';
  text[1] = isupper(fmt.type) ? 'X' : 'x';
  print_hex((size_t)obj, 16, isupper(fmt.type), text + 2);
  stream_text(fmt, '>', text, 18, s);
}

template<typename type_t>
auto make_class_string(const type_t& obj);

template<typename type_t>
auto make_map_string(const type_t& obj);

template<typename type_t>
auto make_array_string(const fmt_t& fmt, const type_t& obj);

template<typename type_t>
auto stream_generic(fmt_t fmt, const type_t& obj, std::string& s) {
  if constexpr(requires { static_cast<const char*>(obj); }) {
    // Has a decay to const char*. Prefer this over std::string, because it's
    // probably faster.
    const char* p = static_cast<const char*>(obj);
    stream_text(fmt, '<', p, strlen(p), s);

  } else if constexpr(std::is_integral_v<type_t>) {
    // Use 'd' as the default integral format type.
    fmt.type = 'd';
    stream_integer(fmt, obj, s);

  } else if constexpr(std::is_floating_point_v<type_t>) {
    // Use 'g' as the default floating-point format type.
    fmt.type = 'g';
    stream_floating(fmt, obj, s);

  } else if constexpr(std::is_pointer_v<type_t>) {
    // Use 'p' as the default pointer type.
    fmt.type = 'p';
    stream_pointer(fmt, obj, s);

  } else if constexpr(requires { static_cast<std::string>(obj); }) {
    // The type has a user-defined conversion to string. Use this.
    std::string s2 = static_cast<std::string>(obj);
    stream_text(fmt, '<', s2.c_str(), s2.size(), s);

  } else if constexpr(std::is_member_pointer_v<type_t>) {
    // No implementation for pointer-to-member just yet.
    @meta std::string msg = va_format(
      "cannot stringify pointer-to-member type %s",
      @type_string(type_t, true)
    );
    @meta throw std::runtime_error(std::move(msg));

  } else if constexpr(std::is_union_v<type_t>) {
    // No implementation for union.
    @meta std::string msg = va_format(
      "cannot stringify union type %s",
      @type_string(type_t, true)
    );
    @meta throw std::runtime_error(std::move(msg));

  } else if constexpr(std::is_enum_v<type_t>) {
    if(const char* name = enum_to_name(obj)) {
      // Stream the name of the enumerator.
      stream_text(fmt, '<', name, strlen(name), s);

    } else {
      // No matching enumerator for this value. Format as an integer.
      fmt.type = 'd';
      stream_integer(fmt, std::underlying_type_t<type_t>(obj), s);
    }

  } else if constexpr(std::is_array_v<type_t> || 
    type_t.template == std::vector ||
    type_t.template == std::array ||
    type_t.template == std::list ||
    type_t.template == std::set ||
    type_t.template == std::multiset) {

    std::string s2 = make_array_string(fmt, obj);
    s += s2;

  } else if constexpr(type_t.template == std::map) {
    std::string s2 = make_map_string(obj);
    stream_text(fmt, '<', s2.c_str(), s2.size(), s);

  } else if constexpr(type_t.template == std::optional) {
    if(obj) {
      // Recurse on the value.
      stream_generic(fmt, *obj, s);

    } else {
      // Stream (null)
      stream_text(fmt, '<', "(null)", 6, s);
    }

  } else if constexpr(std::is_class_v<type_t>) {
    std::string s2 = make_class_string(obj);
    stream_text(fmt, '<', s2.c_str(), s2.size(), s);

  } else {
    @meta std::string msg = va_format(
      "cannot stringify unimplemented type %s",
      @type_string(type_t, true)
    );
    @meta throw std::runtime_error(std::move(msg));
  }
}

template<typename type_t>
auto make_class_string(const type_t& obj) {
  static_assert(std::is_class_v<type_t>);

  // Reserve 16 bytes per data member.
  std::string s;
  s.reserve(16 * @member_count(type_t));

  s += '{';
  @meta for(int i = 0; i < @member_count(type_t); ++i) {
    if(i)
      s += ',';
    s += ' ';

    // Stream the name of the member.
    s += @member_name(type_t, i);

    // Stream a colon name/value separator.
    s += " : ";

    // Stream the member value.
    stream_generic({ }, @member_value(obj, i), s);
  }
  s += " }";

  return s;
}

template<typename type_t>
auto make_map_string(const type_t& obj) {
  // Reserve 16 bytes per data member.
  std::string s;
  s.reserve(16 * std::size(obj));

  s += '{';
  bool insert_comma = false;

  for(auto& x : obj) {
    if(insert_comma)
      s += ',';
    s += ' ';

    // Stream the key.
    stream_generic({ }, x.first, s);

    // Stream the key/value separator.
    s += " : ";

    // Stream the value.
    stream_generic({ }, x.second, s);

    // On the next go-around, insert a comma before the space.
    insert_comma = true;
  }

  s += " ]";
  return s;
}

template<typename type_t>
auto make_array_string(const fmt_t& fmt, const type_t& obj) {
  // Reserve 16 bytes per data member.
  std::string s;
  s.reserve(16 * std::size(obj));

  s += '[';
  bool insert_comma = false;

  for(auto& x : obj) {
    if(insert_comma)
      s += ',';
    s += ' ';

    // Stream the element.
    stream_generic(fmt, x, s);

    // On the next go-around, insert a comma before the space.
    insert_comma = true;
  }

  s += " ]";
  return s;
}

template<int ord, format_type_t fmt_type, int index, typename type_t>
auto stream_arg(arg_t<fmt_type, index, type_t> arg, std::string& s) {

  // Handle array types specially. This will preserve formatting options and 
  // apply them to elements.

  @meta if((std::is_array_v<type_t> && 
      !requires { static_cast<const char*>(arg.obj); }) || 
    type_t.template == std::array ||
    type_t.template == std::vector ||
    type_t.template == std::list ||
    type_t.template == std::set ||
    type_t.template == std::multiset) {

    // Reserve 16 bytes per data member.
    s.reserve(s.capacity() + 16 * std::size(arg.obj));

    s += '[';
    bool insert_comma = false;

    for(auto& x : arg.obj) {
      if(insert_comma)
        s += ',';
      s += ' ';

      // Stream the element.
      stream_arg<ord>(
        make_arg<fmt_type, index>(arg.fmt, arg.fmt.width, arg.fmt.precision, x), 
        s
      );

      // On the next go-around, insert a comma before the space.
      insert_comma = true;
    }

    s += " ]";

  } else @meta+ if(format_type_int == fmt_type) {
    // Error if the object is neither integral nor convertible to long.
    if(!std::is_integral_v<type_t> && !requires { static_cast<long>(arg); }) {
      std::string msg = va_format(
        "integer type expected in argument %d\n"
        "provided argument has type %s",
        ord, @type_string(type_t, true)
      );
      throw std::runtime_error(std::move(msg));      
    }

    @emit stream_integer(arg.fmt, arg.obj, s);

  } else if(format_type_float == fmt_type) {
    // Error if the object is neither float nor convertible to float.
    if(!std::is_floating_point_v<type_t> && !requires { static_cast<double>(arg); }) {
      std::string msg = va_format(
        "floating-point type expected in argument %d\n"
        "provided argument has type %s",
        ord, @type_string(type_t, true)
      );
      throw std::runtime_error(std::move(msg));
    }

    @emit stream_floating(arg.fmt, arg.obj, s);

  } else if(format_type_pointer == fmt_type) {
    if(!std::is_pointer_v<type_t> && !requires { static_cast<const void*>(arg); }) {
      std::string msg = va_format(
        "pointer type expected in argument %d\n"
        "provided argument has type %s",
        ord, @type_string(type_t, true)
      );
      throw std::runtime_error(std::move(msg));
    }

    @emit stream_pointer(arg.fmt, arg.obj, s);

  } else {
    // Stream without a presentation type.
    try {
      @emit stream_generic(arg.fmt, arg.obj, s);

    } catch(std::exception& e) {
      // Catch a compile-time exception and add details.
      std::string msg = va_format("error in format operand %d\n%s\n", ord,
        e.what());
      throw std::runtime_error(std::move(msg));
    }
  }
}


template<int ord, format_type_t fmt_type, int index, typename... types_t>
auto stream_arg(pack_t<fmt_type, index, types_t...> pack, std::string& s) {
  // Manually expand the pack into individual arguments.
  s += '[';
  @meta for(size_t i = 0; i < sizeof...(types_t); ++i) {
    // Stream a comma between pack elements.
    @meta if(i)
      s += ',';
    s += ' ';

    // Stream the pack element as an arg.
    stream_arg<ord>(make_arg<fmt_type, index>(pack.fmt, pack.@(i)), s);
  }

  s += " ]";
}

template<size_t len, typename... args_t>
auto format_text(const char* fmt, const args_t&... args) {
  // Reserve enough space for twice the format text plus 8 bytes per argument.
  std::string s;
  s.reserve(2 * len + 8 * sizeof...(args_t));

  @meta size_t cur = 0;

  // Loop over each pack argument.
  @meta for(size_t i = 0; i < sizeof...(args_t); ++i) {
    // Stream any text preceding this argument.
    @meta size_t end = decltype(args...[i])::index;
    @meta if(cur < end) {
      s.insert(s.end(), fmt + cur, fmt + end);
      @meta cur = end; 
    }

    // Stream this argument's text.
    stream_arg<i>(args...[i], s);
  }

  // Insert text after the last argument.
  @meta if(cur < len)
    s.insert(s.end(), fmt + cur, fmt + len);

  return s;
}

@mauto eval_integer(int index, const std::string& s) {
  @meta if(s.size()) {
    // Eval a width or precision specifier. Convert it to int.
    return (int)@@expression(s);

  } else {
    return index;
  }
}

@mauto eval_arg(const format_arg_t& arg) {
  @meta if(arg.pack) {
    return cirfmt::make_pack<arg.type, arg.index>( 
      arg.fmt,
      eval_integer(arg.fmt.width, arg.width_expr),
      eval_integer(arg.fmt.precision, arg.precision_expr),
      @@expression(arg.expr) ...
    ); 

  } else {
    return cirfmt::make_arg<arg.type, arg.index>( 
      arg.fmt,
      eval_integer(arg.fmt.width, arg.width_expr),
      eval_integer(arg.fmt.precision, arg.precision_expr),
      @@expression(arg.expr)
    ); 
  }
}

@mauto format(const char* fmt) {
  @meta try {
    @meta fmt_builder_t builder { fmt };
    @meta builder.parse();

    return format_text<builder.format2.size()>(
      @string(builder.format2),
      eval_arg(@pack_nontype(builder.args))...
    );

  } catch(...) {
    // Catch and rethrow. This clears out the backtrace, leaving a cleaner
    // error message.
    @meta throw;
  }
}

inline size_t write_f(FILE* f, const std::string& s) {
  return ::fwrite(s.c_str(), 1, s.size(), f);
}

@mauto write(FILE* f, const char* fmt) {
  write_f(f, format(fmt));
}

@mauto write(const char* fmt) {
  return write_f(::stdout, format(fmt));
}


} // namespace cirfmt

// User-defined literal functions for yielding a std::string...
@mauto operator""_format(const char* fmt, size_t) {
  @meta try {
    return cirfmt::format(fmt);

  } catch(...) {
    @meta throw;
  }
}

// ... and for printing to stdout.
@mauto operator""_print(const char* fmt, size_t) {
  @meta try {
    return cirfmt::write(fmt);

  } catch(...) {
    @meta throw;
  }
}

