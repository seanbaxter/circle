#include <vector>
#include <cstdio>

int main() {
  std::vector<int> v { 3, 1, 2, 5, 3, 4, 4, 7, 6 };

  // For each odd number in v, repeat that number that many times.
  // Prints '3 3 3 1 5 5 5 5 5 3 3 3 7 7 7 7 7 7 7'
  std::vector<int> v2 = [for i : v if 1 & i => for x : i => i... ... ];
  printf("%d ", v2[:])...; printf("\n");

  // Cut off the same comprehension after 10 elements.
  // Prints '3 3 3 1 5 5 5 5 5 3'
  std::vector<int> v3 = [for i : v if 1 & i => for x : i => i... ... ] | 10;
  printf("%d ", v3[:])...; printf("\n");

  // Interleave each element of v3 with 0.
  std::vector<int> v4 = [ { v3[:], 0 }... ];
  printf("%d ", v4[:])...; printf("\n");
  
  // Create a triangular structure of vectors.
  auto v5 = [ for i : 5 => [ for x : i => i ... ] ... ];
  for(auto& v : v5) {
    printf("[ "); printf("%d ", v[:])...; printf("]\n");
  }
}