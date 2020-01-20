#include <vector>
#include <cstdio>

int main() {
  std::vector v = [ for i : @range(1:6)... => for i2 : i => i ... ... ];

  // Prints '1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 '.
  printf("%d ", v[:])...; printf("\n");
}
