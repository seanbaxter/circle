#include <algorithm>
#include <cstdio>

int main() {
  int x = 10;
  std::vector<int> v { 5, 3, 1, 4, 2, 3, 5, 1 };
  std::vector<int> u { 7, 1, 2, 2, 4, 3, 8, 7 };

  // 1) Print x * v for each element in v. Use std::for_each.
  std::for_each(v.begin(), v.end(), [=](int y) { printf("%2d ", x * y); });
  printf("\n");

  // Print x * v for each element in v using Circle dynamic packs.
  printf("%2d ", x * v[:])...; printf("\n");

  // 2) Print x * v in reverse order.
  std::for_each(v.rbegin(), v.rend(), [=](int y) { printf("%2d ", x * y); });
  printf("\n");

  // Use a step of -1 to visit the elements in reverse order.
  printf("%2d ", x * v[::-1])...; printf("\n");

  // 3) Use any_of to confirm a number greater than 4.
  bool is_greater = std::any_of(v.begin(), v.end(), [](int x) { return x > 4; });
  printf("STL: greater than 4? %s\n", is_greater ? "true" : "false");

  // Use Circle dynamic packs to confirm a number greater than 4.
  bool is_greater2 = (... || (v[:] > 4));
  printf("Circle: greater than 4? %s\n", is_greater2 ? "true" : "false");

  // 4) Print u * v. How do we do this with STL algorithms? Do we need
  // boost::zip_iterator to present two simultaneous views as one?

  // With Circle, just use two slices.
  printf("%2d ", u[:] * v[:])...; printf("\n");

  // 5) Print the sum of odds and evens. That is, print v[0] + v[1], 
  // v[2] + v[3], etc.
  // Do we need to combine a step_iterator with a zip_iterator? What is the
  // C++ answer?

  // With Circle, use the step argument of two slices.
  printf("%2d ", v[::2] + v[1::2])...; printf("\n");
}