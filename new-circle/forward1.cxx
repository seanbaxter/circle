#pragma feature forward
#include <iostream>

void consume(forward auto... args) {
  std::cout<< "  " + decltype(forward args)~string ...;
  std::cout<< "\n";
}

void func(forward auto pair) {
  // Use the forward-operator on a forwarding parameter to get the right
  // value category. This is a primary-expression, even though it comes on
  // the left. It applies to the parameter, not the subobject. Member-access
  // does the right thing here, propagating the value category of the parameter
  // to its subobjects.
  consume(forward pair);
  consume(forward pair.first, forward pair.second);
}

template<typename T1, typename T2>
struct pair_t {
  T1 first;
  T2 second;
};

int main() {
  std::cout<< "Pass by lvalue:\n";
  pair_t pair { 100, 200.2 };
  func(pair);

  std::cout<< "Pass by rvalue:\n";
  func(pair_t { 1i8, 2ui16 });
}
