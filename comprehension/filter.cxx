#include <vector>
#include <string>
#include <cstdio>

int main() {
  std::vector<int> vi { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

  // Create a vector of strings. Filter for the even elements and convert 
  // to strings.
  std::vector v2 = [for i : vi if 0==i%2 => std::to_string(i) ...];

  // Print the strings out.
  printf("%s ", v2[:].c_str())...; printf("\n"); // Prints '2 4 6 8 10 '.
}
