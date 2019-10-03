#include "json.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

@macro void inject_function(std::string name, std::string text) {
  @meta std::cout<< "Injecting "<< name<< " = "<< text<< "\n";

  // Expand this function definition into the scope from which the macro
  // is called. It could be namespace or class scope.
  double @(name)(double x) {
    return @expression(text);
  }
}

@macro void inject_from_json(const char* filename) {
  // Load a JSON file at compile time. These objects have automatic storage
  // duration at compile time. They'll destruct when the end of the macro
  // is hit.
  @meta std::ifstream inject_file("inject.json");
  @meta nlohmann::json inject_json;
  @meta inject_file>> inject_json;

  // Loop over each item in the file and inject a function.
  @meta for(auto& item : inject_json.items()) {
    @meta std::string key = item.key();
    @meta std::string value = item.value();

    // Expand the inject_function 
    @macro inject_function(key, value);
  }  
}

int main() {
  // Expand this macro into the injected namespace. This creates the namespace
  // if it isn't already created.
  @macro namespace(injected) inject_from_json("inject.json");

  std::cout<< "unity(.3) = "<< injected::unity(.3)<< "\n";

  return 0;
}