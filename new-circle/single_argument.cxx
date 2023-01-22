#pragma feature template_brackets
#include <vector>
#include <string>

// Use the full template-parameter-list form. There is no abbreviated form.
template<int N>
struct obj_t { };

template<typename X>
void func(int i) { }

int main() {
  // Use an abbreviated form of template-argument-list.

  // 1. ! id-expression
  std::vector!std::string v1;

  // 2. ! simple-type-name
  std::vector!int v2;

  // 3. ! a single token
  obj_t!100 v3;

  // This works well for calling function templates.
  func!float(10);
}