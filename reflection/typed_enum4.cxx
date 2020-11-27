#include <type_traits>
#include <algorithm>
#include <cstdio>

enum typename typelist_t {
  int,
  double,
  char[4],
  float,
  double(double),
  float,
  void*,
};

// Check if a type is in the list.
template<typename type_t, typename list_t>
constexpr bool is_type_in_list_v =  
  (... || std::is_same_v<type_t, @enum_types(list_t)>);

// Use + to get a count of the occurrences for a type in the list.
template<typename type_t, typename list_t>
constexpr size_t occurence_in_list_v =
  (... + (size_t)std::is_same_v<type_t, @enum_types(list_t)>);

// True if the list has no duplicate types.
template<typename list_t>
constexpr bool is_unique_list = 
  (... && (1 == occurence_in_list_v<@enum_types(list_t), list_t>));

// Check for any type matching the provided type trait.
template<template<typename> class trait_t, typename list_t>
constexpr bool any_trait_in_list_v =  
  (... || trait_t<@enum_types(list_t)>::value);

// Check that all types match the trait.
template<template<typename> class trait_t, typename list_t>
constexpr bool all_trait_in_list_v =  
  (... && trait_t<@enum_types(list_t)>::value);

// Search for the index of the first occurence of type_t in list_t.
template<typename type_t, typename list_t>
constexpr size_t find_first_in_list_v = std::min({ 
  std::is_same_v<type_t, @enum_types(list_t)> ? int... : @enum_count(list_t)... 
});

int main() {
  printf("char* is in typelist_t = %d\n", is_type_in_list_v<char*, typelist_t>);
  printf("float is in typelist_t = %d\n", is_type_in_list_v<float, typelist_t>);

  printf("float is in typelist_t %d times\n", occurence_in_list_v<float, typelist_t>);
  printf("void* is in typelist_t %d times\n", occurence_in_list_v<void*, typelist_t>);  

  printf("typelist_t is a unique list = %d\n", is_unique_list<typelist_t>);

  printf("integral is in typelist_t = %d\n", any_trait_in_list_v<std::is_integral, typelist_t>);
  printf("function is in typelist_t = %d\n", any_trait_in_list_v<std::is_function, typelist_t>);

  printf("typelist_t are all integral = %d\n", all_trait_in_list_v<std::is_integral, typelist_t>);
  printf("typelist_t are all fundamental = %d\n", all_trait_in_list_v<std::is_fundamental, typelist_t>);

  printf("index of first char[4] = %d\n", find_first_in_list_v<char[4], typelist_t>);
  printf("index of first bool = %d\n", find_first_in_list_v<bool, typelist_t>);
}

