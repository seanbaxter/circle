#feature on placeholder_keyword

// You can use a placeholder in parameter-declaration. It's clearer than
// leaving the declarator unnamed.
void f1(int _, double _) {
  // Permit any number of placeholder name objects.
  auto _ = 1;
  auto _ = 2.2;

  // Error: object must be automatic duration. We need a non-placeholder name
  // for name mangling static duration objects.
  static auto _ = 3;

  // Error: '_' is not an expression.
  func(_);
}

// Works with [new_decl_syntax] too.
#feature on new_decl_syntax

// [new_decl_syntax] requires parameter names, so we must use placeholders
// if we want them unnamed.
fn f2(_ : int, _ : double) {
  // Permit any number of placeholder name objects.
  var _ := 1;
  var _ := 2.2;

  // Error: object must be automatic duration. We need a non-placeholder name
  // for name mangling static duration objects.
  var static _ := 3;
}
