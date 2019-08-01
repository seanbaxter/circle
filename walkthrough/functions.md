# Walkthrough 1: Injecting functions from text

Circle has mechanisms to help you separate the _logic_ of your application from the _code_ of your application. The compiler rotates C++ from the runtime to the compile-time axis, turning C++ into its own scripting language. Use this script/build system capability to turn the translation unit into an automation controller which opens the logical asset, adds some domain-specific intelligence to it, and integrates it into the application.

The first walkthrough keeps it simple. We'll inject function definitions from text:
1. Text stored in an array in the source file.
1. Text stored in a JSON file.
1. Scrape all the JSON files in a directory and inject functions from each of them.

C++ provides unrestricted access to the host environment at runtime. Circle, which is a rotation of C++, provides unrestricted access at compile time. We write C++ code to scrape a directory and load key-value pairs from JSON files, but execute it at compile time to automate code generation.

## Functions from text

[**functions1.cxx**](functions1.cxx)

```cpp
struct func_t {
  const char* name;
  const char* expr;
};

@meta func_t funcs[] {
  { "F1", "(x + y) / x" },
  { "F2", "2 * x * sin(y)" }
};

@meta for(func_t f : funcs) {
  double @(f.name)(double x, double y) {
    return @expression(f.expr);
  }
}
```

Let's create an array of `func_t` and initialize the items with the names and expressions of a couple of functions to turn into code. Mark it `@meta` to indicate that it is only used at compile time--this array will not be emitted into the executable. 

Write a compile-time ranged-for over the items. Now simply write a function definition. The `@()` operator turns strings and integers into identifiers, so our functions will be named `F1` and `F2`. `@expression` tokenizes, parses and injects text as an _expression_, so our functions will return `(x + y) / x` and `2 * x * sin(y)`. 

But in what declaration region are the functions actually declared? Because they aren't prefixed with `@meta`, the functions are _real_ declarations. The block scope established by the meta for is a _meta_ scope. The function is declared in this meta scope, so that it's in the declarative region of `f.name` and `f.expr` -- it needs access to that object. At the end of each iteration, we hit the end of the meta for's block scope, and during its cleanup it destructs all meta objects and drops real declarations into the containing scope, which in this case is the global namespace. In effect, real declarations in meta scopes fall through into the innermost enclosing real scope (like namespaces, class and enum definitions, and function block scopes).

```cpp
typedef double(*fp_t)(double, double);
fp_t get_func_by_name(const char* name) {
  @meta for(func_t f : funcs) {
    if(!strcmp(@string(f.name), name))
      return ::@(f.name);
  }
  return nullptr;
}

int main(int argc, char** argv) {
  if(4 != argc) {
    fprintf(stderr, "usage is %s [func-name] [x] [y]\n", argv[0]);
    return 1;
  }

  fp_t fp = get_func_by_name(argv[1]);
  if(!fp) {
    fprintf(stderr, "%s is not a recognized function\n", argv[1]);
    return 1;
  }
  
  double x = atof(argv[2]);
  double y = atof(argv[3]);

  double result = fp(x, y);
  printf("result is %f\n", result);
  return 0;
}
```

Let's make a second pass over the `funcs` array and generate a function that matches a function name to a function pointer. We want this to work at runtime, so we'll emit a sequence of `strcmp` calls. The `@string` extension takes any string known at compile time (in this case, `f.name`, which is a const char\*), and turns it into a string literal. If the string compare succeeds, we lookup `::@(f.name)`, yielding the lvalue of the function, and converting it to a function pointer during its implicit conversion in the return statement. 

Why generate functions like this? Because we put the _logic_ of the functions into a single location: the `funcs` array. If we add, remove or modify functions in this array, all supporting operations are automatically updated when the source is recompiled.

## Separating the logic from the source

[**test.json**](test.json)
```json
{
  "F1" : "(x + y) / x",
  "F2" : "2 * x * sin(y)"
}
```

Let's move the functions out of the .cxx file and into a .json file. This is easier to keep track of, easier to test, easier to share with other tools and languages. Imagine the transformations you can apply on this logical resource... Documentation generation, accuracy testing, derivative generation, power series construction and so on. Think about the resource as expression your intent. You run it through your pipeline and add intelligence. Then you use Circle as a scripting language to integrate it into your C++ program.

JSON bindings exist for every language. For C++, we'll use the single-header version of the popular [nlohmann parser](https://github.com/nlohmann/json/blob/develop/single_include/nlohmann/json.hpp).

[**functions2.cxx**](functions2.cxx)
```cpp
#include "json.hpp"

// Keep an array of the names of functions we injected.
@meta std::vector<std::string> func_names;

// Inject a function given a name and return-statement expression.
@macro void inject_f(const char* name, const char* expr) {
  @meta std::cout<< "Injecting function '"<< name<< "'\n";

  double @(name)(double x, double y) {
    return @expression(expr);
  }

  // Note that we added this function.
  @meta func_names.push_back(name);
}

// Open a JSON from a filename and loop over its items. Inject each item
// with inject_f.
@macro void inject_from_json(const char* filename) {
  // Open the file and read it as JSON
  @meta std::ifstream f(filename);
  @meta nlohmann::json j;
  @meta f>> j;

  // Loop over all items in the JSON.
  @meta for(auto& item : j.items()) {
    @meta std::string name = item.key();
    @meta std::string value = item.value();

    // Inject each function.
    @macro inject_f(name.c_str(), value.c_str());
  }
}

// Inject a file's worth of functions.
@macro inject_from_json("test.json");

// Map a function name to the function lvalue.
typedef double(*fp_t)(double, double);
fp_t get_func_by_name(const char* name) {
  @meta for(const std::string& s : func_names) {
    if(!strcmp(@string(s), name))
      return ::@(s);
  }

  return nullptr;
}
```
```
$ circle functions2.cxx
Injecting function 'F1'
Injecting function 'F2'
```

This program introduces Circle statement macros. Circle macros are a lot like functions: they exist in the normal declarative regions, participate in argument deduction (when you write macro templates) and overload resolution (when name lookup finds multiple functions/macros). The main difference is that function definitions expand into their own scope, whereas macros expand into the scope where they're called. If you expand a macro from inside a class definition, the macro's contents are inserted into the class definition itself.

The `inject_f` macro simply declares a two-parameter function given a compile-time name and expression string. We'll print the name of the injected function as a diagnostic. Additionally, we'll push the name of the function to a compile-time `vector<string>`, so that we can later generate a function to map names to function pointers.

`inject_from_function` does the real work. Given a filename, it uses iostreams to open a file and initialize the JSON parser with the file's contents. This is all done at compile time! The json.hpp code is parsed and injected as AST, like any other library, but here we'll evaluate it inside Circle's integrated interpreter rather than emitting it as LLVM IR. 

A compile-time range-for loops over each entry in the JSON and expands the `inject_f` macro, declaring a functions in the global namespace. (The global namespace is still the innermost real scope at this point--all the other braces come after meta constructions.)

## Circle as a build system

Since Circle provides access to the host environment at compile time, we can use operating system services to better automate our builds. Let's throw another JSON file into the source directory:

[**test.json**](test.json)
```json
{
  "F1" : "(x + y) / x",
  "F2" : "2 * x * sin(y)"
}
```

[**test2.json**](test2.json)
```json
{
  "F3" : "sqrt(x * x + y * y)",
  "F4" : "(x > y) ? x : y"
}
```

The source file can scrape the source directory (or, more realistically, a resource directory), read in all the JSON files, and generate code from their contents.

[**functions3.cxx**](functions3.cxx)
```cpp
#include <dirent.h>

inline std::string get_extension(const std::string& filename) {  
  return filename.substr(filename.find_last_of(".") + 1);
}

inline bool match_extension(const char* filename, const char* ext) {
  return ext == get_extension(filename);
}

@macro void inject_from_dir(const char* dirname) {
  // Get a cursor into the indicated directory.
  @meta DIR* dir = opendir(dirname);

  // Loop over all files.
  @meta while(dirent* ent = readdir(dir)) {
    // Match .json files.
    @meta if(match_extension(ent->d_name, "json")) {

      // Inject all the functions named in this JSON file.
      @macro inject_from_json(ent->d_name);
    }
  }

  // Close the resource.
  @meta closedir(dir);
}

// Inject a file's worth of functions.
@macro inject_from_dir(".");
```
```
$ circle functions3.cxx
Injecting function 'F3'
Injecting function 'F4'
Injecting function 'F1'
Injecting function 'F2'
```

`inject_from_dir` is the new entry point for code generation. It uses the POSIX API [opendir](https://pubs.opengroup.org/onlinepubs/009695399/functions/opendir.html) to create a cursor into the contents of a directory. Each call to [readdir](https://pubs.opengroup.org/onlinepubs/009695399/functions/readdir.html) returns a descriptor of the pointed-at item in the directory, and advances the cursor to the next item. The final call to [closedir](https://pubs.opengroup.org/onlinepubs/009695399/functions/closedir.html) is an unnecessary but thoughtful act of civic responsibility.

After matching the file's extension, we simply expand the `inject_from_json` macro on that filename. We're now inside an _if-statement_ inside a _while-statement_, but these are all meta constructs. The innermost real scope is still the global namespace, so those functions are injected there. (However there exist sneaky mechanisms for injecting a declaration into any namespace from any scope, if you find yourself in that situation.)

Why invest in all this scraping of directories? It allows us to evolve the program's logical assets and its source code independently. Separation of code from data is a decades-old truism. Artists creating textures for a game aren't declaring each new contribution into the source code. There's a critical separation of concerns. In Circle, thanks to its C++-as-a-script capability, one can separate logical assets from source code too.
