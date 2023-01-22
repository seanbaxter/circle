#include <string>
#include <iostream>

struct Data {
  // Strings in tickmarks 
  std::string `First Name`;
  std::string `Last Name`;
  int single, `double`, triple;
};

int main() {
  Data data { };
  data.`First Name` = "Abe";
  data.`Last Name` = "Lincoln";
  data.single = 1;
  data.`double` = 2;
  data.triple = 3;

  // Use reflection to print the name of each member and its value.
  std::cout<< Data~member_names + ": "<< data~member_values<< "\n" ...;
}