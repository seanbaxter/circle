#pragma feature self

struct foo_t {
  void func() {
    self.x += 10;    // OK!
    this->x += 10;   // Error
  }

  // Take an explicit object by using the 'self' parameter name.
  template<typename T>
  void func2(T self) {
    self.x = 1;
  }
  
  int x = 10;
};