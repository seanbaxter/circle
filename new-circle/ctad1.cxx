#include <vector>

int main() {
  std::vector v1 { 1, 2, 3, 4 };

  // This is bad - does not copy the elements from v1. It simply
  // inserts the begin and end iterators.
  std::vector v2 { v1.begin(), v1.end() };

  // Disable CTAD.
  #pragma feature no_class_template_argument_deduction

  // Errors with an indicator of the deduced specialization. The user can 
  // check if it's what they intended and then insert it by hand.
  std::vector v3 { v1.begin(), v1.end() };

  // Copy-deduction guides are still okay, because they can't be misused.
  std::vector v4 = v1; // OK!
}