#include <vector>
#include <iostream>

int main() {
  using std::cout;

  std::vector<int> v { 6, 2, 3, 4, 5, 6 };

  cout<< std::boolalpha;
  cout<< "vector any_of is 6: "<< (... || (6 == v[:]))<< '\n';
  cout<< "vector all_of is 6: "<< (... && (6 == v[:]))<< '\n';
  cout<< "vector none_of is 6: "<< (... && (6 != v[:]))<< '\n';
}