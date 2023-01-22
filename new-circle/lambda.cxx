#include <cstdio>

int main() {
  // Define a lambda that takes two template arguments.
  auto f = []<typename T, typename U>() {
    puts("T = {}, U = {}".format(T~string, U~string));
  };

  // Call it the old way.
  f.template operator()<int, double>();

  // Call it the new way.
  #pragma feature template_brackets
  f!<int, double>();
}