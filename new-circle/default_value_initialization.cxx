#pragma feature default_value_initialization

// Has a trivial default constructor, so it gets value-initialized.
struct foo_t { int x, y, z, w; };

// Has a non-trivial default constructor, so that gets called instead.
struct bar_t { bar_t() { } int x, y; };

int main() {
  int a;                     // Value-initialized to 0.
  int foo_t::*b;             // Value-initialized to -1.
  int (foo_t::*c)();         // Value-initialized to 0.
  foo_t d;                   // Value-initialized to 0.
  bar_t e;                   // bar_t::bar_t is executed.
   
  int f           = void;    // Uninitialized.
  int foo_t::*g   = void;    // Uninitialized.
  int (foo_t::*h) = void;    // Uninitialized.
  foo_t i         = void;    // Uninitialized.    
  // bar_t j         = void; // Error! bar_t must have a trivial default constructor.
}