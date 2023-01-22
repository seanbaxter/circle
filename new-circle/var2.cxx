#pragma feature new_decl_syntax
#include <cstdio>

struct S {
  var x : int;
  var y : int;
  var z : int;
};

int main() {
  var [x, y, z] := S { 1, 2, 3 };
  printf("%d %d %d\n", x, y, z);
}