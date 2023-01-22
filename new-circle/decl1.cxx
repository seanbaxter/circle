#pragma feature new_decl_syntax choice

// Look, these curly-brace definitions don't need a trailing ';'.
// That's because [new_decl_syntax] requires types be declared/defined 
// in their own declaration statements.


struct foo_t;
struct foo_t { }

enum class my_enum_t {
  a, b, c,
}

choice choice_t { 
  x(int),
}

// Can still use elaborated-type-specifier, but not at the start of a 
// declaration.
fn func() -> struct foo_t;

// Valued-initialized object.
var x : int;

// Use type inference for double.
var y := 101.1;

// Error: [new_decl_syntax]: typedef is disabled; use an alias-declaration
// typedef int x;

fn main() -> int {
  var y := 10;
}