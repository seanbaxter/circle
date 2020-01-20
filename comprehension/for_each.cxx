#include <vector>
#include <array>
#include <list>
#include <forward_list>
#include <queue>
#include <iostream>

int main() {
  using std::cout;

  cout<< "vector:   ";
  std::vector<int> v { 1, 2, 3, 4, 5, 6 };
  cout<< v[:]<< ' ' ...;

  cout<< "\narray:    ";
  std::array<int, 6> a { 1, 2, 3, 4, 5, 6 };
  cout<< a[:]<< ' ' ...;

  cout<< "\nlist:     ";
  std::list<int> ll { 1, 2, 3, 4, 5, 6 };
  cout<< ll[:]<< ' ' ...;

  cout<< "\nfwd_list: ";
  std::forward_list<int> fl { 1, 2, 3, 4, 5, 6 };
  cout<< fl[:]<< ' ' ...;

  cout<< "\ndeque:    ";
  std::deque<int> d { 1, 2, 3, 4, 5, 6 };
  cout<< d[:]<< ' ' ...;
  cout<< '\n';
}