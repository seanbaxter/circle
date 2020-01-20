#include <vector>
#include <array>
#include <iostream>

int main() {
  using std::cout;
  cout<< std::boolalpha;

  std::vector<int> v { 1, 2, 3, 4, 5, 6 };
  bool is_sorted1 = (... && (v[:] <= v[1:]));
  cout<< "vector: "<< is_sorted1<< '\n';

  std::array<int, 6> a { 6, 2, 3, 4, 5, 6 };
  bool is_sorted2 = (... && (a[:] <= a[1:]));
  cout<< "array:  "<< is_sorted2<< '\n';
}
