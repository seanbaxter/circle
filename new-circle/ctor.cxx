#pragma feature no_user_defined_ctors

struct foo_t {
  foo_t() = default;                 // OK.
  foo_t(const foo_t&) = delete;      // OK.
  foo_t(int x);                      // Prohibited.
};

#pragma feature new_decl_syntax
struct bar_t {
  fn bar_t() = default;              // OK.
  fn bar_t(_:const foo_t&) = delete; // OK
  fn bar_t(x: int);                  // Prohibited.
};