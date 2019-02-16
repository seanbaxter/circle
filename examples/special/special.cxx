#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

// REMINDER: Break the json

// Header-only JSON parser.
#include <json.hpp>

// Parse the JSON file and keep it open in j.
using nlohmann::json;
@meta json j;

// Open a json file.
@meta std::ifstream json_file("special.json");

// Parse the file into the json object j.
@meta json_file>> j;

// Define a square function and expose it to the formulas in the json.
double sq(double x) {
  return x * x;
}

// Loop over each json item and define the corresponding functions.
@meta for(auto& item : j.items()) {
  @meta json& value = item.value();

  // Extract and print the name.
  @meta std::string name = item.key();
  @meta printf("Generating code for function '%s'\n", name.c_str());

  // Print the note if one exists.
  @meta+ if(value.count("note"))
    printf("  note: %s\n", value["note"].get<std::string>().c_str());

  // Define the function as f_XXX.
  extern "C" double @("f_" + name)(double x) {
    @meta if(value.count("integer")) {
      // If the function has an integer flag, the incoming argument must be
      // a positive integer. Do a runtime check and fail if it isn't.
      if(roundf(x) != x || x < 1.0) {
        printf("[ERROR]: Argument to '%s' must be a positive integer.\n",
          @string(name));
        abort();
      }
    }

    @meta if(value.count("f")) {
      // The "f" field has an expression which we want to evaluate and 
      // return.
      @meta std::string f = value["f"].get<std::string>();
      @meta printf("  Injecting expression '%s'\n", f.c_str());

      // Inject the expression tokens and parse as primary-expression. Returns
      // the expression.
      return @expression(f);

    } else @meta if(value.count("statements")) {
      // The "statements" field has a sequence of statements. We'll execute 
      // these and let it take care of its own return statement.
      @meta std::string s = value["statements"].get<std::string>();
      @meta printf("  Injecting statements '%s'\n", s.c_str());

      @statements(s, "special.json:" + name);

    } else {
      // Circle allows static_assert to work on const char* and std::string
      // objects that are known at compile-time.
      static_assert(false,  
        "\'" + name + "\'" + " must have 'f' or 'statements' field");
    }
  }

  // Define the series as series_XXX if a Taylor series exists.
  @meta if(value.count("series")) {
    extern "C" double @("series_" + name)(double x) {
      @meta printf("  Injecting Taylor series\n");

      double xn = 1;
      double y = 0;
      @meta json& terms = value["series"];
      @meta for(double c : terms) {
        @meta if(c)
          y += xn * c;
        xn *= x;
      }

      return y;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

double evaluate(const char* name, bool is_series, double x) {
  @meta for(auto& item : j.items()) {
    // Compile-time iterate over each element of special.json looking
    // for the key that matches name. We cannot use j.find(name), because
    // name is not known until runtime.
    @meta std::string name2 = item.key();

    // Convert the meta std::string object to a string literal so we can 
    // compare at runtime.
    if(0 == strcmp(name, @string(name2))) {
      double y = 0;
      if(is_series) {
        @meta if(item.value().count("series")) {
          y = @("series_" + name2)(x);

        } else {
          printf("[ERROR]: %s has no 'series' defined.\n", name);
          abort();
        }

      } else {
        y = @("f_" + name2)(x);
      }
      return y;
    }
  }

  printf("[ERROR]: The function '%s' is unsupported by this tool.\n", name);
  abort();
}

void print_usage() {
  printf("  Usage: special [series|function] name x\n");
  exit(1);
}

int main(int argc, char** argv) {  
  if(4 != argc)
    print_usage();

  bool is_series = 0 == strcmp(argv[1], "series");
  bool is_function = 0 == strcmp(argv[1], "function");
  if(!is_series && !is_function) 
    print_usage();

  double x = atof(argv[3]);
  double y = evaluate(argv[2], is_series, x);

  printf("%s(%f) = %f\n", argv[2], x, y);
  return 0;
}

