#feature on new_decl_syntax

// Clearer function declaration syntax. Always use trailing-return-type.
fn func1(x: int) -> double { return 3.14; }

// Works like usual with templates.
template<typename T>
fn func2(x: T) -> int {
  return (int)x;
}

// Parameter packs use leading dots.
template<typename... Ts>
fn func3(...args: Ts);

// Or use an invented template parameter pack.
fn func4(...args: auto);

// C-style ellipsis parameters are indicated as usual.
fn func5(p: const char*, ...);

struct obj_t {
  // Special member functions are declared in the typical way, but now have
  // an unambiguous syntax.
  fn obj_t() = default;
  fn ~obj_t() = default;

  // For conversion functions, the return type is implied by the 
  // function name.
  fn operator int() const {
    return 1;
  }

  // Ordinary member functions.
  fn func() const -> int {
    return 100;
  }
}