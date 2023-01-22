#include <iostream>
#include <tuple>

// Use Circle member traits to turn the std::tuple_elements of a tuple-like
// into a pack. Evaluate the concept C on each tuple_element of T.
// They must all evaluate true.
template<typename T, template<typename> concept C>
concept tuple_like_of = (... && C<T~tuple_elements>);

// Constrain func to tuple-like integral types. Note we're passing a concept
// as a template parameter.
void func(tuple_like_of<std::integral> auto tup) { 
  std::cout<< "func called with type {}\n".format(decltype(tup)~string);
}

int main() {
  func(std::tuple<int, unsigned short>());    // OK
  func(std::array<char, 5>());                // OK
  func(std::pair<short, char>());             // OK
  // func(std::tuple<int, const char*>());    // Error
}