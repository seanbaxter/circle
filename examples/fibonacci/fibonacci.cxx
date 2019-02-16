#include <cstdio>
#include <cstdlib>
#include <vector>

// An ordinary function.
std::vector<int> fib(int count) {
  std::vector<int> vec(count);
  vec[0] = 0;
  vec[1] = 1;
  for(int i = 2; i < count; ++i)
    vec[i] = vec[i - 2] + vec[i - 1];
  return vec;
}

// Another ordinary function.
void print_numbers(int count) {
  std::vector<int> vec = fib(count);
  for(int i = 0; i < vec.size(); ++i)
    printf("%3d: %8d\n", i, vec[i]);
}

int main(int argc, char** argv) {
  // Parse Fibonacci number count at runtime.
  int count = atoi(argv[1]);

  // Call ordinary function at runtime.
  print_numbers(count);

  // Call externally-defined functions at compile-time.
  @meta printf("How many numbers? (Must be >= 2)\n  ");
  @meta int count2 = 2;
  @meta scanf("%d", &count2);

  // Call ordinary function at compile-time.
  @meta print_numbers(count2);
  return 0;
}