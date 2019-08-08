#pragma once
#include <cstdarg>
#include <type_traits>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>

template<typename type_t>
const char* name_from_enum(type_t e) {
  static_assert(std::is_enum<type_t>::value);
  
  switch(e) {
    @meta for enum(type_t e2 : type_t) {
      // @enum_value is the i'th unique enumerator in type_t.
      // eg, circle, square, rhombus
      case e2:
        // @enum_name returns a string literal of the enumerator.
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

template<typename type_t>
std::optional<type_t> enum_from_name(const char* name) {
  @meta for enum(type_t e : type_t) {
    if(!strcmp(@enum_name(e), name))
      return e;
  }

  return { };
}

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

template<typename type_t>
void stream_simple(std::ostream& os, const type_t& obj) {

  if constexpr(std::is_enum<type_t>::value) {
    // For the simple stream, just write the enumerator name, not the 
    // enumeration type.
    if(const char* name = name_from_enum<type_t>(obj)) {
      // Write the enumerator name if the value maps to an enumerator.
      os<< name;
      
    } else {
      // Otherwise cast the enum to its underlying type and write that.
      os<< (typename std::underlying_type<type_t>::type)obj;
    }

  } else if constexpr(@is_class_template(type_t, std::basic_string)) {
    // For the simple case, stream the string without quotes. This is closer
    // to ordinary printf behavior.
    os<< obj;

  } else if constexpr(std::is_same<const char*, typename std::decay<type_t>::type>::value) {
    os<< obj;

  } else if constexpr(
    std::is_array<type_t>::value || 
    @is_class_template(type_t, std::vector) ||
    @is_class_template(type_t, std::set) ||
    @is_class_template(type_t, std::multiset)) {

    // Special treatment for std::vector. Output each element in a comma-
    // separated list in brackets.
    os<< '[';
    bool insert_comma = false;
    for(const auto& x : obj) {
      // Move to the next line and indent.
      if(insert_comma)
        os<< ',';
      os<< ' ';
      
      // Stream the element.
      stream_simple(os, x);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< " ]";

  } else if constexpr(@is_class_template(type_t, std::map)) {
    // Special treatment for std::map.
    os<< '{';
    bool insert_comma = false;
    for(const auto& x : obj) {
      if(insert_comma)
        os<< ",";
      os<< ' ';

      // stream the key.
      stream_simple(os, x.first);

      os<< " : ";

      // stream the value.
      stream_simple(os, x.second);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }  
    os<< " }";

  } else if constexpr(@is_class_template(type_t, std::optional)) {
    // For an optional member, either stream the value or stream "null".
    if(obj)
      stream_simple(os, *obj);
    else
      os<< "null";

  } else if constexpr(std::is_class<type_t>::value) {
    // For any other class, treat with circle's introspection.
    os<< '{';
    bool insert_comma = false;
    @meta for(size_t i = 0; i < @member_count(type_t); ++i) {
      if(insert_comma) 
        os<< ",";
      os<< ' ';

      // Stream the name of the member. The type will be prefixed before the
      // value.
      os<< @member_name(type_t, i)<< " : ";

      // Stream the value of the member.
      stream_simple(os, @member_ref(obj, i));

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< " }";

  } else {
    // For any non-class type, use the iostream overloads.
    os<< obj;
  }
}

template<typename type_t>
std::string cir_to_string(const type_t& obj) {
  std::ostringstream oss;
  stream_simple(oss, obj);
  return oss.str();
}

////////////////////////////////////////////////////////////////////////////////

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

@macro auto esprintf(@meta const char* fmt) {
  // Process the input specifier. Remove {name} and replace with %s.
  // Store the names in the array.
  @meta std::vector<std::string> names;
  @meta std::string fmt2;
  @meta transform_format(fmt, fmt2, names);

  // Convert each name to an expression and from that to a string.
  // Pass to sprintf via format.
  return format(
    @string(fmt2.c_str()), 
    cir_to_string(@expression(@pack_nontype(names))).c_str()...
  );
}

@macro auto eprintf(const char* fmt) {
  return std::cout<< esprintf(fmt);
}

@macro auto operator ""_e(const char* fmt, size_t len) {
  return esprintf(fmt);
}

