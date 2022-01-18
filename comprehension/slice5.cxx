#include <vector>
#include <array>
#include <iostream>

template<typename type_t>
void print_dynamic(const type_t& obj) {
  std::cout<< "[ ";

  // A homogeneous print operation that uses dynamic pack expansion
  // and generates a runtime loop. The container must implement .begin()
  // and .end().
  std::cout<< obj[:]<< " "...;

  std::cout<< "]\n";
}

template<typename type_t>
void print_static(const type_t& obj) {
  std::cout<< "[ ";

  // A heterogenous print operation. Uses static pack expansion. Works on
  // member objects of class types, regular arrays, plus types implementing
  // std::tuple_size, such as std::array, std::pair and std::tuple.
  std::cout<< obj.[:]<< " "...;

  std::cout<< "]\n";
}

int main() {
  std::array<int, 8> array { 1, 2, 3, 4, 5, 6, 7, 8 };

  // Dynamic pack indexing generates a loop.
  array[:] *= 2 ...;
  print_dynamic(array);

  // Static pack indexing performs template substitution to unroll
  // the operation.
  ++array.[:] ...;
  print_static(array);

  // Use list comprehension to generate an std::vector.
  // Expansion of dynamic slice creates a dynamic loop.
  std::vector v1 = [ array[:] * 3... ];
  print_dynamic(v1);

  // Expansion of static slice expansion occurs during substitution.
  // This supports using heterogeneous containers as initializers in
  // list comprehension and uniform initializers.
  std::vector v2 = [ array.[:] * 4... ];
  print_dynamic(v2);

  // Use static slice expansion to create an initializer list for a 
  // builtin array. This won't work with the dynamic slice operator [:], 
  // because the braced initializer must have a compile-time set number of 
  // elements.
  int array2[] { array.[:] * 5 ... };
  print_static(array2);

  // Create a braced initializer in forward then reverse order.
  int forward_reverse[] { array.[:] ..., array.[::-1] ...};
  print_static(forward_reverse);

  // Create a braced initializer with evens then odds.
  int parity[] { array.[0::2] ..., array.[1::2] ... };
  print_static(parity);

  // Use a compile-time loop to add up all elements of array3.
  int static_sum = (... + array.[:]);
  printf("static sum = %d\n", static_sum);

  // Use a dynamic loop to add up all elements of array3.
  int dynamic_sum = (... + array[:]);
  printf("dynamic sum = %d\n", dynamic_sum);
}
