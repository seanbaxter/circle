#include <cstdio>
#include <variant>
#include <string>

template<typename type_t>
void func(type_t var) {
  @match(var) {
    <{ int         }> 5            => printf("It's 5\n");
    <{ int         }> 1 ... 10 _x  => printf("1 <= %d < 10\n", _x);
    <{ int         }>              => printf("Some other int\n");
    
    <{ std::string }> "Hello"     => printf("It says Hello!\n");
    <{ std::string }> "Goodbye"   => printf("It says Goodbye!\n");
    <{ std::string }>             => printf("Some other string\n");
 
    <{ double      }> 0 ... 3.14  => printf("A double in the range of pi\n");
    <{ double      }> < 0 _x      => _x = abs(_x);  // Make it positive.
    <{ double      }>             => printf("Some other double\n");
  };
}

int main() {
  std::variant<int, std::string, double> var;

  var = 4;
  func(var);

  var = "Goodbye";
  func(var);

  var = 1.1;
  func(var);

  return 0;
}