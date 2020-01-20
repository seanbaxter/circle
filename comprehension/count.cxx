#include <vector>
#include <array>
#include <iostream>

int main() {
  using std::cout;

  // Count the number of 6s.
  // Promote each comparison to int, because adding bools in a dynamic
  // fold expression will return a bool type.
  std::vector<int> v { 6, 2, 3, 4, 5, 6 };
  int count1 = (... + (int)(6 == v[:]));
  cout<< "vector: "<< count1<< '\n';

  // Do it with an array.
  std::array<int, 6> a { 6, 2, 3, 4, 5, 6 };
  int count2 = (...+ (int)(6 == a[:]));
  cout<< "array: "<< count2<< '\n';
}
