#include <utility>
#include <tuple>
#include <iostream>

// Print comma-separated arguments.
template<typename... types_t>
void print_args(types_t... args) {
  std::cout<< "[ ";
  
  // Use static slicing to print all but the last element, followed by
  // a comma.
  std::cout<< args...[:-2]<< ", "...;

  // Use direct indexing to print the last element, if there is one.
  // The static subscript ...[-1] refers to the last element.
  // ...[-2] refers to the second-to-last element, and so on.
  if constexpr(sizeof...(args) > 0)
    std::cout<< args...[-1]<< " ";

  std::cout<< "]\n";
}

template<int... values>
void print_nontype() {
  std::cout<< "[ ";
  
  // Print template non-type arguments the same way you print function args.
  std::cout<< values...[:-2]<< ", "...;
  
  // Print the last non-type argument.
  if constexpr(sizeof...(values) > 0)
    std::cout<< values...[-1]<< " ";

  std::cout<< "]\n";
}

int main() {
  auto tuple = std::make_tuple('A', 1, 2.222, "Three");

  // Use static indexing to turn a tuple into a parameter pack. Expand it
  // into function arguments.
  std::cout<< "tuple to pack in forward order:\n";
  print_args(tuple...[:] ...);

  // Or expand it in reverse order.
  std::cout<< "\ntuple to pack in reverse order:\n";
  print_args(tuple...[::-1] ...);

  // Or send the even then the odd elements.
  std::cout<< "\neven then odd tuple elements:\n";
  print_args(tuple...[0::2] ..., tuple...[1::2] ...);

  // Pass indices manually to a template.
  std::cout<< "\ntemplate non-type arguments sent the old way:\n";
  print_nontype<3, 4, 5, 6>();

  // Or use static slicing to turn an array, class or tuple-like object
  // into a parameter pack and expand that into a template-arguments-list.
  std::cout<< "\ntemplate non-type arguments expanded from an array:\n";
  constexpr int values[] { 7, 8, 9, 10 };
  print_nontype<values...[:] ...>();
}