void func(const int*);

int main() {
  func(nullptr);  // OK
  func(0);        // OK

  #pragma feature no_zero_nullptr
  func(nullptr);  // OK
  func(0);        // Error
}