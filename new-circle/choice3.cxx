#feature on choice new_decl_syntax
#include <iostream>
#include <concepts>

fn even(x : std::integral auto) noexcept -> bool {
  return 0 == x % 2;
}

fn func(arg : auto) {
  match(arg) {
    5                => std::cout<< "The arg is five.\n";
    10 ... 20        => std::cout<< "The arg is between 10 and 20.\n";
    even             => std::cout<< "The arg is even.\n";
    1 || 3 || 7 || 9 => std::cout<< "The arg is special.\n";
    _                => std::cout<< "The arg is not special.\n";
  };
}

fn main() -> int {
  func(5);
  func(13);
  func(32);
  func(7);
  func(21);
}