#feature on edition_carbon_2023
#include <string>
#include <iostream>

using String = std::string;

choice IntResult {
  Success(int32_t),
  Failure(String),
  Cancelled,
}

fn ParseAsInt(s: String) -> IntResult {
  var r : int32_t = 0;
  for(var c in s) {
    if(not isdigit(c)) {
      return .Failure("Invalid character");
    }

    // Accumulate the digits as they come in.
    r = 10 * r + c - '0';
  }

  return .Success(r);
}

fn TryIt(s: String) {
  var result := ParseAsInt(s);
  match(result) {
    .Success(var x)   => std::cout<< "Read integer "<< x<< "\n";
    .Failure(var err) => std::cout<< "Failure '"<< err<< "'\n";
    .Cancelled        => std::terminate();
  };
}

fn main() -> int {
  TryIt("12345");
  TryIt("12x45");
  return 0;
}
