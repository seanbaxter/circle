#include <vector>
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

enum typename typelist_t {
  int,
  double,
  char[4],
  int,
  char[4],
  void*,
};

// Create a collection of the unique types. Use an order-preserving unique
// function.
enum typename unique_list_t {
  // Convert each type in joined_list_t to an @mtype. @mtype is a builtin
  // type that encapsulates a type and has comparison/relational operators
  // defined. You can sort or unique with it.
  @meta std::vector types { 
    @dynamic_type(@enum_types(typelist_t)) ... 
  };

  // Create a unique set of @mtypes.
  @meta stable_unique(types);

  // Convert all the unique types into enumerator declarations.
  @pack_type(types)...;
};

// We can also sort the type lexicographically by their string representations.
enum typename sorted_list_t {
  // Expand the string spellings of the types into an array, along with the
  // index into the type.
  @meta std::array types {
    std::make_pair<std::string, int>( 
      @enum_type_strings(typelist_t),
      int...
    )...
  };

  // Lexicographically sort the types.
  @meta std::sort(types.begin(), types.end());

  // Gather the types and define enumerators.
  @enum_type(typelist_t, @pack_nontype(types).second) ...;
};

int main() {
  printf("unique_list_t:\n");
  printf("  %2d: %s\n", int..., @enum_type_strings(unique_list_t))...;

  printf("\nsorted_list_t:\n");
  printf("  %2d: %s\n", int..., @enum_type_strings(sorted_list_t))...;
}