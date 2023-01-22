#pragma feature choice
#include <iostream>

choice IntResult {
  Success(int),
  Failure(std::string),
  Cancelled,
};

template<typename T>
void func(T obj) {
  match(obj) {
    .Success(auto x) => std::cout<< "Success: "<< x<< "\n";
    .Failure(auto x) => std::cout<< "Failure: "<< x<< "\n";
    .Cancelled       => std::terminate();
  };
}

int main() {
  IntResult r1 = .Success(12345);
  IntResult r2 = .Failure("Hello");
  IntResult r3 = .Cancelled();
  func(r1);
  func(r2);
  func(r3);
}