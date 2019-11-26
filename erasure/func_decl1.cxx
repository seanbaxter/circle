#include <cstdio>
#include <cmath>

struct func_decl_t {
  const char* type;
  const char* name;
  const char* expr;
};

@meta const func_decl_t func_decls[] {
  {
    "int(int, int)",
    "absdiff",
    "abs(args...[1] - args...[0])"
  },
  {
    "double(double)",
    "sq",
    "args...[0] * args...[0]"
  }
};

// Loop over the entries in func_decl_t at compile time.
@meta for(func_decl_t decl : func_decls) {

  @meta const char* expr = decl.expr;

  // Declare a function for each entry.
  @func_decl(@type_id(decl.type), decl.name, args) {

    // Return the expression.
    return @expression(decl.expr);
  }
}

int main() {
  int x = absdiff(5, 7);
  printf("absdiff(5, 7) -> %d\n", x);

  double y = sq(3.14159);
  printf("sq(3.14158) -> %f\n", y);

  return 0;
}