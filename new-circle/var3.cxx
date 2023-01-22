#pragma feature new_decl_syntax
#include <cstdio>

int main() {
  var array : int[] { 1, 2, 3, 4 };
  var sum := 0;

  for(var x in array) {
    printf("%d\n", x);
    sum += x;
  }

  printf("%d\n", sum);
}