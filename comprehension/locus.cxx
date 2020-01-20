#include <vector>
#include <cstdio>

int main() {
  
  // Create a vector of vectors, but use different expansion loci.
  auto m1 = [ [ @range(5)... ] ];
  auto m2 = [ [ @range(5) ]... ];

  printf("m1:\n[\n");
  for(auto& v : m1) {
    printf("  [ "); printf("%d ", v[:])...; printf("]\n");
  }
  printf("]\n\n");

  printf("m2:\n[\n");
  for(auto& v : m2) {
    printf("  [ "); printf("%d ", v[:])...; printf("]\n");
  }
  printf("]\n");
}
