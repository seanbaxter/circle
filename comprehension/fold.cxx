#include <vector>
#include <algorithm>
#include <cstdio>

inline int fact(int x) {
  // Use a fold expression to compute factorials. This evaluates the product
  // of integers from 1 to x, inclusive.
  return (... * @range(1:x+1));
}

inline void func() {
  std::vector<int> v { 4, 2, 2, 2, 5, 1, 1, 9, 8, 7, 1, 7, 4, 1 };
  
  // (... || pack) is a short-circuit fold on operator||.
  bool has_five = (... || (5 == v[:]));
  printf("has_five = %s\n", has_five ? "true" : "false");

  bool has_three = (... || (3 == v[:]));
  printf("has_three = %s\n", has_three ? "true" : "false");

  // Reduce the number of 1s.
  int num_ones = (... + (int)(1 == v[:]));
  printf("has %d ones\n", num_ones);

  // Find the max element using qualified lookup for std::max.
  int max_element = (... std::max v[:]);
  printf("max element = %d\n", max_element);

  // Find the min element using the ADL trick. This uses unqualified lookup
  // for min.
  using std::min;
  int min_element = (... min v[:]);
  printf("min element = %d\n", min_element);

  // Find the biggest difference between consecutive elements.
  int max_diff = (... std::max (abs(v[:] - v[1:])));
  printf("max difference = %d\n", max_diff);

  // Compute the Taylor series for sign. s is the current index, so
  // pow(-1, s) alternates between +1 and -1.
  // The if clause in the for-expression filters out the even elements, 
  // where are zero for sine, and leaves the odd powers. This compacts the
  // vector to 5 elements out of 10 terms.
  int terms = 10;
  std::vector series = [for i : terms if 1 & i => pow(-1, i/2) / fact(i)...];
  printf("series:\n");
  printf("  %f\n", series[:])...;

  // Compute x raised to each odd power. Use @range to generate all odd 
  // integers from 1 to terms, and raise x by that.
  double x = .3;
  std::vector powers = [pow(x, @range(1:terms:2))...];
  printf("powers:\n");
  printf("  %f\n", powers[:])...;

  // Evaluate the series to approximate sine. This is a simple dot
  // product between the coefficient and the powers vectors.
  double sinx = (... + (series[:] * powers[:]));
  printf("sin(%f) == %f\n", x, sinx);
}

@meta func();

int main() {
  std::vector<int> v { 4, 2, 2, 2, 5, 1, 1, 9, 8, 7, 1, 7, 4, 1 };
  
  // (... || pack) is a short-circuit fold on operator||.
  bool has_five = (... || (5 == v[:]));
  printf("has_five = %s\n", has_five ? "true" : "false");

  bool has_three = (... || (3 == v[:]));
  printf("has_three = %s\n", has_three ? "true" : "false");

  // Reduce the number of 1s.
  int num_ones = (... + (int)(1 == v[:]));
  printf("has %d ones\n", num_ones);

  // Find the max element using qualified lookup for std::max.
  int max_element = (... std::max v[:]);
  printf("max element = %d\n", max_element);

  // Find the min element using the ADL trick. This uses unqualified lookup
  // for min.
  using std::min;
  int min_element = (... min v[:]);
  printf("min element = %d\n", min_element);

  // Find the biggest difference between consecutive elements.
  int max_diff = (... std::max (abs(v[:] - v[1:])));
  printf("max difference = %d\n", max_diff);

  // Compute the Taylor series for sign. s is the current index, so
  // pow(-1, s) alternates between +1 and -1.
  // The if clause in the for-expression filters out the even elements, 
  // where are zero for sine, and leaves the odd powers. This compacts the
  // vector to 5 elements out of 10 terms.
  int terms = 10;
  std::vector series = [for i : terms if 1 & i => pow(-1, i/2) / fact(i)...];
  printf("series:\n");
  printf("  %f\n", series[:])...;

  // Compute x raised to each odd power. Use @range to generate all odd 
  // integers from 1 to terms, and raise x by that.
  double x = .3;
  std::vector powers = [pow(x, @range(1:terms:2))...];
  printf("powers:\n");
  printf("  %f\n", powers[:])...;

  // Evaluate the series to approximate sine. This is a simple dot
  // product between the coefficient and the powers vectors.
  double sinx = (... + (series[:] * powers[:]));
  printf("sin(%f) == %f\n", x, sinx);
}

