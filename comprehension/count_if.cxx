#include <vector>
#include <array>
#include <iostream>

int main() {
  using std::cout;

  // This is the identical implementation as count.cxx. 
  // ranges::count and ranges::count_if are implemented with the same
  // fold expression in Circle.
  std::vector<int> v { 6, 2, 3, 4, 5, 6 };
  int count1 = (... + (int)(6 == v[:]));
  cout<< "vector: "<< count1<< '\n';

  std::array<int, 6> a { 6, 2, 3, 4, 5, 6 };
  int count2 = (...+ (int)(6 == a[:]));
  cout<< "array: "<< count2<< '\n';
}
