#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

template<int... indices>
struct foo_t { 
  @meta printf("foo_t instantiated on ( ");
  @meta printf("%d ", indices)...;
  @meta printf(")\n");
};

int main() {
  // Create an uninitialized array of integer values.
  @meta int values[10];

  // Use expression pack expansion to set each element of values. The length
  // of the array is deduced from the type.
  @meta @pack_nontype(values) = rand() % 100 ...;

  // Sort the indices.
  @meta std::sort(values, values + 10);

  // Instantiate foo_t on the sorted indices.
  foo_t<@pack_nontype(values)...> my_foo;

  return 0;
}