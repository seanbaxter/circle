#pragma feature choice tuple new_decl_syntax
#include <tuple>
#include <iostream>

struct obj_t {
  var a : (int, int);                  // A 2-tuple.
  var b : (std::string, double, int);  // A 3-tuple.
}

fn func(arg : auto) {
  match(arg) {
    // Destructure the a member and test if it's (10, 20)
    [a: (10, 20)]             => std::cout<< "The 'a' member is (10, 20).\n";

    // Check the range of the double tuple element.
    [_, [_,  1...100, _] ]   => std::cout<< "The double is between 1 and 100\n";

    // a's 2nd element matches b's third element.
    [ [_, var x], [_, _, x] ] => std::cout<< "A magical coincidence.\n";

    // Everything else goes here.
    _                         => std::cout<< "A rubbish struct.\n";
  };
}

fn main() -> int {
  func(obj_t { { 10, 20 }, { "Hello", 3, 4     } });
  func(obj_t { { 2, 4 },   { "Hello", 3, 4     } });
  func(obj_t { { 2, 5 },   { "Hello", 19.0, 4  } });
  func(obj_t { { 2, 5 },   { "Hello", 101.0, 5 } });
  func(obj_t { { 2, 5 },   { "Hello", 101.0, 6 } });
}