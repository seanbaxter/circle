#feature on choice new_decl_syntax
#include <string>
#include <tuple>
#include <iostream>

choice MyChoice {
  MyTuple(int, std::string),   // The payload is a tuple.
  MyArray(double[3]),          // The payload is an array.
  MyScalar(short),             // The payload is a scalar.
}

fn test(obj : MyChoice) {
  // You can pattern match on tuples and arrays.
  match(obj) {
    .MyTuple([1, var b])    => std::cout<< "The tuple int is 1\n";
    .MyArray([var a, a, a]) => std::cout<< "All array elements are "<< a<< "\n";
    .MyArray(var [a, b, c]) => std::cout<< "Some other array\n";
    .MyScalar(> 10)         => std::cout<< "A scalar greater than 10\n";
    .MyScalar(var x)        => std::cout<< "The scalar is "<< x<< "\n";
    _                       => std::cout<< "Something else\n";
  };
}

fn main() -> int {
  test(.MyTuple{1, "Hello choice tuple"});
  test(.MyArray{10, 20, 30});
  test(.MyArray{50, 50, 50});
  test(.MyScalar{100});
  test(.MyScalar{6});
  test(.MyTuple{2, "Foo"});
}