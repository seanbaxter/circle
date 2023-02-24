void func(const int*);

int main() {
  func(nullptr);  // OK
  func(0);        // OK

  #feature on no_zero_nullptr
  func(nullptr);  // OK
  func(0);        // Error
}