#include <string>

// In C/C++ you must use identifier names.
int x;

// Circle provides dynamic names. Enclose a string or integer in @().
int @("y");                       // Name with a string literal.

using namespace std::literals;    // Name with an std::string.
int @("z"s + std::to_string(1));  // Same as 'int z1;'

int @(5);                         // Same as 'int Z5'

// Declares int _100, _101, _102, _103 and _104 in global namespace.
@meta for(int i = 100; i < 105; ++i)
  int @(i);

int main() {
  _102 = 1;

  return 0;
}
