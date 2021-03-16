#include <vector>
#include <algorithm>
#include <iostream>

template<typename type_t>
void stable_unique(std::vector<type_t>& vec) {
  // std::sort + std::unique also works, but isn't stable.
  std::vector<type_t> vec2;
  for(type_t x : vec) {
    if(vec2.end() == std::find(vec2.begin(), vec2.end(), x))
      vec2.push_back(x);
  }
  vec = std::move(vec2);
}

template<typename... types_t>
struct tuple_t {
  // Loop over all members of the parameter pack.
  // Declare a non-static data member for each pack element.
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};


template<typename... types_t>
struct unique_tuple_t {
  // Create a compile-time std::vector<@mtype>. @mtype encapsulates a 
  // type, allowing you to manipulate it like a variable. This means we can
  // sort them! @dynamic_type converts a type to an @mtype prvalue.
  @meta std::vector<@mtype> types { @dynamic_type(types_t)... };

  // Use an ordinary function to unique these types.
  @meta stable_unique(types);

  // @pack_type returns an array/std::vector<@mtype> as a type parameter pack.
  // Print the unique list of names as a diagnostic.
  @meta std::cout<< @type_string(@pack_type(types))<< "\n"...;

  // Typedef a tuple_t over these unique types.
  typedef tuple_t<@pack_type(types)...> type_t;
};

int main() {
  // Create a tuple of just the unique types in the arguments list.
  // This has only four data members.
  typename unique_tuple_t<int, double, char*, double, char*, float>::type_t 
    tuple { };

  // Prints tuple_t<int, double, char*, float>
  std::cout<< @type_string(decltype(tuple), true)<< "\n";

  return 0;
}