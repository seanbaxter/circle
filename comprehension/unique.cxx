#include <iostream>
#include <vector>
#include <algorithm>

int main() {

  std::vector<int> v{9, 4, 5, 2, 9, 1, 0, 2, 6, 7, 4, 5, 6, 5, 9, 2, 7,
                      1, 4, 5, 3, 8, 5, 0, 2, 9, 3, 7, 5, 7, 5, 5, 6, 1,
                      4, 3, 1, 8, 4, 0, 7, 8, 8, 2, 6, 5, 3, 4, 5};

  std::sort(v.begin(), v.end());

  // Use list comprehension to create a unique vector.
  // Always emit the first element, since it has to be unique.
  // Then loop over all remaining elements, and compare with the subsequent
  // element. If they differ, emit the subsequent element.
  auto v2 = [v[0], for i : v.size() - 1 if v[i] != v[i+1] => v[i+1] ...];

  std::cout<< v2[:]<< " " ...;
  std::cout<< "\n";


}