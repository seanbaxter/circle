#include "json.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

// Load a JSON file at compile time.
@meta std::ifstream inject_file("inject.json");
@meta nlohmann::json inject_json;
@meta inject_file>> inject_json;

// Loop over each item in the file and inject a function.
@meta for(auto& item : inject_json.items()) {
  @meta std::string key = item.key();
  @meta std::string value = item.value();

  // Print a diagnostic
  @meta std::cout<< "Injecting "<< key<< " = "<< value<< "\n";

  // Define a function with the key name into the global namespace.
  double @(key)(double x) {
    // Convert the value text into an expression.
    return @expression(value);
  }
}

int main() {
  std::cout<< "unity(.3) = "<< unity(.3)<< "\n";

  return 0;
}