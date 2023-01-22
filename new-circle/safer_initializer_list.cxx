#include <vector>

int main() {
  std::vector<int> v1(4, 4);   // OK, [4, 4, 4, 4]
  std::vector<int> v2{4, 4};   // OK, [4, 4] - Surprising!

  #pragma feature safer_initializer_list
  std::vector<int> v3(4, 4);   // OK, [4, 4, 4, 4]
  std::vector<int> v4{4, 4};   // ERROR, Ambiguous initialization
  std::vector<int> v5{{4, 4}}; // OK, [4, 4]
}