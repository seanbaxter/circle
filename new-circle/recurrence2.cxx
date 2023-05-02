#feature on recurrence
#include <vector>
#include <iostream>

// P2355 - Postfix fold expressions
// decltype(auto) index(auto &arr, auto... ii) {
//   return (arr[...][ii]);
// }

// Circle's recurrence relation.
decltype(auto) index(auto& arr, auto... ii) {
  return (recurrence arr[ii] ...);
}

int main() {
  using std::vector;
  vector<vector<vector<vector<int>>>> data;
  data.resize(4);
  data[1].resize(4);
  data[1][3].resize(4);
  data[1][3][2].resize(4);
  data[1][3][2][3] = 100;

  // Perform an n-dimensional index to get our value out.
  std::cout<< index(data, 1, 3, 2, 3)<< "\n";
}