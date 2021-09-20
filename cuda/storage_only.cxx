#include <iostream>

struct foo_t {
  // A deleted default ctor!
  foo_t() = delete;

  // A defined one-parameter ctor.
  foo_t(int x) : x(x) { 
    std::cout<< "Constructor called\n";
  }

  ~foo_t() {
    std::cout<< "Destructor called\n";
  }

  int x;
};

// Define foo without calling its ctor or dtor. Init/destruction is
// up to the user.
[[storage_only]] foo_t foo;

int main() {
  std::cout<< "Entered main\n";

  // Initialize the object.
  new (&foo) foo_t(100);

  std::cout<< foo.x<< "\n";

  // Destruct the object.
  foo.~foo_t();

  std::cout<< "Exiting main\n";
}