#pragma once
#include <cstdio>
#include <type_traits>
#include <cstdarg>

using namespace std::string_literals;

inline void print(int indent, const char* pattern, ...) {
  va_list args;
  va_start(args, pattern);

  while(indent--)
    printf("  ");
  
  std::vprintf(pattern, args);
}

template<typename enum_t>
std::optional<const char*> enum_to_string(enum_t e) {
  switch(e) {
    @meta for enum(enum_t e2 : enum_t) {
      case e2:
        return @enum_name(e2);
    }

    default:
      return { };
  }
}

// Recursively print a class object's data.
template<typename type_t>
void print_object(const type_t& obj, int indent = 0) {
  if constexpr(std::is_same_v<std::string, type_t>) {
    print(indent, "\"%s\"\n", obj.c_str());

  } else if constexpr(std::is_same_v<const char*, type_t>) {
    print(indent, "\"%s\"\n", obj);

  } else if constexpr(std::is_class_v<type_t>) {
    // Raise a compiler error for protected/private bases.
    static_assert(
      !@base_count(type_t, protected private),
      "cannot stream type \""s + @type_string(type_t) + 
        "\" with non-public bases"
    );

    // Raise a compiler error for protected/private data members.
    static_assert(
      !@member_count(type_t, protected private),
      "cannot stream type \""s + @type_string(type_t) + 
        "\" with non-public member objects"
    );

    // Loop over base classes and recurse.
    const int num_bases = @base_count(type_t);
    @meta for(int i = 0; i < num_bases; ++i) {
      print(indent, "base %s:\n", @base_type_string(type_t, i));
      print_object(obj.@base_value(i), indent + 1);
    }

    // Loop over data members and recurse.
    const int num_members = @member_count(type_t);
    @meta for(int i = 0; i < num_members; ++i) {
      print(indent, "%s:\n", @member_decl_string(type_t, i));
      print_object(obj.@member_value(i), indent + 1);
    }

  } else if constexpr(std::is_array_v<type_t>) {
    const int extent = std::extent_v<type_t>;
    for(int i = 0; i < extent; ++i) {
      print(indent, "[%d]:\n", i);
      print_object(obj[i], indent + 2);
    }

  } else if constexpr(__is_vector(type_t)) {
    const int size = __vector_size(type_t);
    for(int i = 0; i < size; ++i) {
      print(indent, "[%d]:\n", i);
      print_object(obj[i], indent + 2);
    }

  } else if constexpr(std::is_enum_v<type_t>) {
    if(auto s = enum_to_string(obj))
      print(indent, "%s\n", *s);
    else
      print(indent, "<unknown enum %s: %d\n", @type_string(type_t), (int)obj);

  } else if constexpr(std::is_same_v<bool, type_t>) {
    print(indent, "%s\n", obj ? "true" : "false");

  } else if constexpr(std::is_floating_point_v<type_t>) {
    print(indent, "%f\n", obj);

  } else {
    // If the type isn't integral, raise a compiler error.
    static_assert(
      std::is_integral_v<type_t>, 
      "type \""s + @type_string(type_t) + "\" not supported"
    );

    print(indent, "%lld\n", (int64_t)obj);
  }
}