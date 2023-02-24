#feature on new_decl_syntax placeholder_keyword
#include <iostream>

// Standard syntax.
var x0 : int;         // Default-initalization.
var x1 : int = 100;   // Copy-initialization.

// Nice syntax for type inference.
var y := 10;          // Copy-initialization.

// Put the decl-specifier-seq in the usual place.
var static z := 5;

// You can use a braced-initializer for types that support it.
var array : int[] { 1, 2, 3 };

// We don't need 'var' for function parameter declarations. That's assumed.
fn foo(x : int, y : double) -> int { return 1; }

// Get a function pointer. You have to give the function parameter a name, 
// but you can use the placeholder _.
// The declarator syntax hasn't been redesigned, so you still need a base type
// in function types.
var fp1 : int(*)(_ : int, _ : double) = &foo;
var fp2 : auto(*)(_ : int, _ : double)->int = &foo;
var fp3 := &foo;   // Use type inference

struct foo_t {
  var x : int;

  // Put the storage-class-specifier right after 'var'.
  var static y : double;
}

// Use var-declaration for non-type template parameters.
template<var A : int, var B : int>
fn func();

template<typename... Ts>
struct tuple {
  // A member pack declaration. Use leading ...
  var ...m : Ts;
}

fn main()->int {
  // Use var for declaring init-statement variables for loops.
  for(var i := 0; i < 5; ++i)
    std::cout<< "for: "<< i<< "\n";

  // Use var with 'in' for ranged-for statements. This replaces the ':'
  // in Standard C++.
  for(var i in 5)
    std::cout<< "ranged for: "<< i<< "\n";

  // Use var for condition objects in if-statements.
  if(var i := y * y)
    std::cout<< "if: "<< i<< "\n";
}