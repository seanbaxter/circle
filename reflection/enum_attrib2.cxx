#include <cstdio>

using name [[attribute]] = const char*;

enum [[.name="an enumeration"]] foo_t {
  a [[.name="an enumerator"]]
};

[[.name="an object"]] const foo_t x = a;  

int main() {
  puts(@attribute(foo_t, name));         // prints "an enumeration"
  puts(@attribute(a, name));             // prints "an enumerator"
  puts(@attribute(x, name));             // prints "an object"
  puts(@attribute(decltype(x), name));   // prints "an enumeration"
  puts(@enum_attribute(x, name));        // prints "an enumerator"
}