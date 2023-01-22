#pragma feature template_brackets
#include <iostream>

template!<typename... Ts>
void func(Ts... args) {
  // Print all types.
  std::cout<< Ts~string<< " " ...;
  std::cout<< "\n";
}

void dispatch(auto f) {
  // Provide two template arguments. This may follow template arguments
  // specified where the 
  f!<double, short>(1, 2, 3, 4);
}

int main() {
  // It works with overload sets containing at least one function template.
  dispatch([]func);

  // We can optionally provied the starting template arguments here.
  // At the call site, additional template arguments are appended.
  dispatch([]func!<float>);
}

