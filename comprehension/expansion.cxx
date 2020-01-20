#include <vector>
#include <set>
#include <algorithm>
#include <cstdio>

int main() {
  std::vector<int> v { 4, 2, 2, 2, 5, 1, 1, 9, 8, 7 };

  // Print the vector values.
  printf("%3d ", v[:])...; printf("\n");

  // Fill the vector with powers of 2.
  v[:] = 1<< @range()...;
  printf("%3d ", v[:])...; printf("\n");

  // Populate a set with the same values. Print their values.
  // Slice is like an enhanced ranged-for, so it supports the usual
  // STL containers, or anything else with begin, end and size member
  // functions.
  std::set<int> set;
  set.insert(v[:])...;
  printf("%3d ", set[:])...; printf("\n");

  // Add up all values into an accumulator. This is better done with a
  // fold expression.
  int sum = 0;
  sum += v[:]...;
  printf("sum = %d\n", sum); // sum = 41

  // Add each right element into its left element. Because the loop is
  // executed left-to-right, we don't risk overwriting any elements before
  // we source them.
  v[:] += v[1:]...;
  printf("%3d ", v[:])...; printf("\n"); //  6  4  4  7  6  2 10 17 15  7 

  // Reset the array to 1s.
  v[:] = 1...;

  // Perform a prefix scan. Add into each element the sum of all elements
  // before it. This is like the fourth example, but with the operands 
  // flipped.
  v[1:] += v[:]...;
  printf("%3d ", v[:])...; printf("\n"); //  1  2  3  4  5  6  7  8  9 10

  // Reverse the array in place. Exchange each element with its mirror
  // up through the midpoint.
  int mid = v.size() / 2;
  std::swap(v[:mid], v[::-1])...;
  printf("%3d ", v[:])...; printf("\n"); // 10  9  8  7  6  5  4  3  2  1

  // Add into each element its index from the range of integers.
  v[:] += @range()...;
  printf("%3d ", v[:])...; printf("\n"); // 10 10 10 10 10 10 10 10 10 10

  // Reset the array to ascending integers. Now swap the even and odd
  // positions. The 2-element step skips every other item.
  v[:] = @range()...;
  std::swap(v[::2], v[1::2])...;
  printf("%3d ", v[:])...; printf("\n"); //  1  0  3  2  5  4  7  6  9  8
}
