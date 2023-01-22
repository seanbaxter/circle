struct a_t { };

struct b_t { 
  b_t(a_t); 
  int x; 
};

int main() {
  // Most vexing parse: This is not an object declaration. It's a function
  // declaration in block scope. a_t() is parsed like a function parameter
  // rather than an initializer.
  b_t obj(a_t());

  // Error: can't access obj.x, because obj isn't an object, it's a 
  // function name.
  obj.x = 1;
}