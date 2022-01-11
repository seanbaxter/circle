#include "variant.hxx"

struct no_compare_t { };

int main() {
  circle::variant<int, long, no_compare_t> a, b;
  bool c = a < b;
}
