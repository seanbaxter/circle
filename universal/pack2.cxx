#include <iostream>

void print_values(const auto&&... x) {
  std::cout<< "{";
  if constexpr(sizeof...(x))
    std::cout<< " "<< x...[0];
  std::cout<< ", "<< x...[1:] ...;
  std::cout<< " }\n";
}

template<int N>
void func() {
  print_values(int...(N)...);
}

int main() {
  // Legacy gcc __integer_pack intrinsic. This is how std::integer_sequence 
  // is actually implemented;
  print_values(__integer_pack(5)...);

  // New int...() expression. It can be given a count...
  print_values(int...(5)...);

  // ... Or it can be given a slice.
  // Print the odds between 1 and 10.
  print_values(int...(1:10:2)...);

  // Print a countdown from 9 to 0. When the step is negative, the 
  // begin index is exclusive and the end index is inclusive.
  print_values(int...(10:0:-1)...);

  struct obj_t {
    int x, y, z;
  };
  obj_t obj { 100, 200, 300 };
  std::cout<< int...(1:)<< ": "<< obj.[:]<< "\n" ...;
}