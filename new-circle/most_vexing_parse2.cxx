#pragma feature new_decl_syntax

struct a_t { }

struct b_t { 
  fn b_t(a : a_t);
  var x : int; 
}

fn main() -> int {
  // Ok. The most vexing parse has been resolved. This is explicitly
  // a variable declaration, not a function declaration.
  var obj : b_t = a_t();

  // Ok. obj really is an object.
  obj.x = 1;
}