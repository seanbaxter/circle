#include <type_traits>
#include <vector>
#include <array>
#include <algorithm>
#include <string>
#include <cstdio>

// A nice order-preserving unique function.
template<typename value_t>
void stable_unique(std::vector<value_t>& vec) {
  auto begin = vec.begin();
  auto end = begin;
  for(value_t& value : vec) {
    if(end == std::find(begin, end, value))
      *end++ = std::move(value);
  }
  vec.resize(end - begin);
}

////////////////////////////////////////////////////////////////////////////////
// Typed enum definitions.

// Define a typed enum.
enum typename my_types1_t {
  double,
  int,
  char*,
  int[5],
  char,
};

// Can access enumerators in a normal loop.
@meta printf("my_types_1:\n");
@meta for(size_t i = 0; i < @enum_count(my_types1_t); ++i)
  @meta printf("  %s\n", @enum_type_string(my_types1_t, i));

// Or access them in a for-enum loop.
@meta printf("my_types_1:\n");
@meta for enum(auto e : my_types1_t)
  @meta printf("  %s\n", @enum_type_string(e));

// But it's more concise to use a pack expansion expression.
@meta printf("my_types_1:\n");
@meta printf("  %s\n", @enum_type_strings(my_types1_t)) ...;

// Define another typed enum.
enum typename my_types2_t {
  char,
  int[5],
  std::nullptr_t,
  char(*)(double),
};
@meta printf("my_types_2:\n");
@meta printf("  %s\n", @enum_type_strings(my_types2_t)) ...;

// Concatenate the typelists above using pack declarations.
enum typename joined_list_t {
  @enum_types(my_types1_t)...;
  @enum_types(my_types2_t)...;
};
@meta printf("joined_list_t:\n");
@meta printf("  %s\n", @enum_type_strings(joined_list_t)) ...;

////////////////////////////////////////////////////////////////////////////////
// Typed enum unique and sorting.

// Create a collection of the unique types. Use an order-preserving unique
// function.
enum typename unique_list_t {
  // Convert each type in joined_list_t to an @mtype. @mtype is a builtin
  // type that encapsulates a type and has comparison/relational operators
  // defined. You can sort or unique with it.
  @meta std::vector<@mtype> types { 
    @dynamic_type(@enum_types(joined_list_t)) ... 
  };

  // Create a unique set of @mtypes.
  @meta stable_unique(types);

  // Convert all the unique types into enumerator declarations.
  @pack_type(types)...;
};
@meta printf("unique_list_t:\n");
@meta printf("  %s\n", @enum_type_strings(unique_list_t)) ...;

// We can also sort the type lexicographically by their string representations.
enum typename sorted_list_t {
  // Expand the string spellings of the types into an array, along with the
  // index into the type.
  @meta std::array types {
    std::make_pair<std::string, int>( 
      @enum_type_strings(unique_list_t),
      int...
    )...
  };

  // Lexicographically sort the types.
  @meta std::sort(types.begin(), types.end());

  // Gather the types and define enumerators.
  @enum_type(unique_list_t, @pack_nontype(types).second) ...;
};
@meta printf("sorted_list_t:\n");
@meta printf("  %s\n", @enum_type_strings(sorted_list_t)) ...;

////////////////////////////////////////////////////////////////////////////////
// Type traits on typed enums.

using my_types_t = unique_list_t;

// Check if a type is in the list.
template<typename type_t, typename list_t>
constexpr bool is_type_in_list_v =  
  (std::is_same_v<type_t, @enum_types(list_t)> || ...);

@meta printf("char* is in my_types_t = %d\n", is_type_in_list_v<char*, my_types_t>);
@meta printf("float is in my_types_t = %d\n", is_type_in_list_v<float, my_types_t>);

// Check for any type matching the provided type trait.
template<template<typename> class trait_t, typename list_t>
constexpr bool any_trait_in_list_v =  
  (trait_t<@enum_types(list_t)>::value || ...);

@meta printf("integral is in my_types_t = %d\n", any_trait_in_list_v<std::is_integral, my_types_t>);
@meta printf("function is in my_types_t = %d\n", any_trait_in_list_v<std::is_function, my_types_t>);

// Check that all types match the trait.
template<template<typename> class trait_t, typename list_t>
constexpr bool all_trait_in_list_v =  
  (trait_t<@enum_types(list_t)>::value && ...);  

@meta printf("my_types_t are all integer = %d\n", all_trait_in_list_v<std::is_integral, my_types_t>);
@meta printf("my_types_t are all fundamental = %d\n", all_trait_in_list_v<std::is_fundamental, my_types_t>);

////////////////////////////////////////////////////////////////////////////////
// Typed enums are complete once the closing brace is hit, and cannot be
// modified after that. If you want a dynamically modifiable set of types,
// use an std::vector<@mtype>. @pack_type on an array, std::array or 
// std::vector of @mtype will yield a type parameter pack.
// Use @dynamic_type and @static_type to convert to and from @mtypes.

@meta std::vector<@mtype> type_vector {
  @dynamic_type(int),
  @dynamic_type(char*),
  @dynamic_type(void(int))
};
@meta printf("type_vector:\n");
@meta printf("  %s\n", @pack_type(type_vector).string) ...;

// Now re-insert these types but as pointers to them. This uses the obscure
// std::vector insert overload
// iterator insert( const_iterator pos, std::initializer_list<T> ilist );
@meta type_vector.insert(
  type_vector.end(),
  { @dynamic_type(@pack_type(type_vector)*) ... }
);
@meta printf("type_vector (2):\n");
@meta printf("  %s\n", @pack_type(type_vector).string) ...;

////////////////////////////////////////////////////////////////////////////////
// Transform between tuples and enums.

// The one-line Circle tuple.
template<typename... types_t>
struct tuple_t {
  // int... is an expression that yields the index of the current element.
  types_t @(int...) ...;
};

// A collection of types.
enum typename tuple_types_t {
  int,
  const char*,
  void(*)(int),
};

// Instantiate a tuple with these types.
typedef tuple_t<@enum_types(tuple_types_t)...> my_tuple_t;

// Print the member decls of the tuple.
@meta printf("my_tuple_t:\n");
@meta printf("  %s\n", @member_decl_strings(my_tuple_t)) ...;

// A collection of names.
enum class tuple_names_t {
  fred,
  barney,
  betty
};

// Combine names and types from two different enums into a single struct.
template<typename types_t, typename names_t>
struct named_tuple_t {
  @enum_types(types_t) @(@enum_names(names_t)) ...;
};

typedef named_tuple_t<tuple_types_t, tuple_names_t> my_named_tuple_t;

// Print the member decls of the tuple.
@meta printf("my_named_tuple_t:\n");
@meta printf("  %s\n", @member_decl_strings(my_named_tuple_t)) ...;
