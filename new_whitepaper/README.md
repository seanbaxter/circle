# The Circle programming language

* Start with C++.
* Integrate an interpreter. 
    * Anything can be run at compile time.
    * Full access to the host environment.
* Reflection.
    * Simply interleave compile-time control flow with regular declarations.
* Include introspection keywords.
    * No runtime cost. Introspection gives access to info already maintained by the compiler.
* Dynamic names.
    * `@()` turns strings and integers into identifiers.
* Liberate parameter packs from templates.
    * Many new extensions return parameter packs.
    * Subscript template parameter packs with `...[]`.
* Injection from text.
    * `@type_id` - a type from a string
    * `@expression` - an expression from a string
    * `@statements` - a sequence of statements from a string
    * `@include` - programmatically `#include` a file from a filename.
* `@mtype` encapsulates a type-id in a type.
    * `@dynamic_type` and `@static_type` convert between types and `@mtype`.
    * Store in standard containers, sort them, unique them. 
* Powerful macros.
    * Circle macros undergo argument deduction and overload resolution like normal functions.
    * Expand their contents into the calling scope.
    * Create a meta context to hold compile-time variables.

## Hello Circle

[**hello.cxx**](hello.cxx)
```cpp
#include <cstdio>

int main() {
  @meta printf("Hello circle\n");
  printf("Hello world\n");
  return 0;
}
```
```
$ circle hello.cxx
Hello circle
$ ./hello
Hello world
```

Put the `@meta` keyword in front of a statement to execute it at compile time. There's an integrated interpreter for executing full functions, and you can make foreign function calls to routines that are defined externally.

## Dynamic names - turn strings and ints into identifiers

[**dynamic_names.cxx**](dynamic_names.cxx)
```cpp
#include <string>

// In C/C++ you must use identifier names.
int x;

// Circle provides dynamic names. Enclose a string or integer in @().
int @("y");                       // Name with a string literal.

using namespace std::literals;    // Name with an std::string.
int @("z"s + std::to_string(1));  // Same as 'int z1;'

int @(5);                         // Same as 'int _5'

// Declares int _100, _101, _102, _103 and _104 in global namespace.
@meta for(int i = 100; i < 105; ++i)
  int @(i);

int main() {
  _102 = 1;   // _102 really declared in global ns.

  return 0;
}
```

Use the Circle extension `@()` to turn a `const char*`, `std::string` or integer into an identifier. Any string known at compile time can be used; not just literals.

## Tuple - a flat data structure

[**tuple.cxx**](tuple.cxx)
```cpp
#include <iostream>

template<typename... types_t>
struct tuple_t {
  // Loop over all members of the parameter pack.
  // Declare a non-static data member for each pack element.
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};

int main() {
  tuple_t<int, double, const char*> my_tuple {
    5, 1.618, "Hello tuple"
  };

  std::cout<< my_tuple._0<< "\n";
  std::cout<< my_tuple._1<< "\n";
  std::cout<< my_tuple._2<< "\n";

  return 0;
}
```
```
$ circle tuple.cxx
$ ./tuple
5
1.618
Hello tuple
```

## Unique types - imperative metaprogramming

[**unique.cxx**](unique.cxx)
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

template<typename... types_t>
struct tuple_t {
  // Loop over all members of the parameter pack.
  // Declare a non-static data member for each pack element.
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};

template<typename type_t>
void make_unique(std::vector<type_t>& vec) {
  // The usual C++ unique trick.
  std::sort(vec.begin(), vec.end());
  vec.resize(std::unique(vec.begin(), vec.end()) - vec.begin());
}

template<typename... types_t>
struct unique_tuple_t {
  // Create a compile-time std::vector<@mtype>. @mtype encapsulates a 
  // type, allowing you to manipulate it like a variable. This means we can
  // sort them! @dynamic_type converts a type to an @mtype prvalue.
  @meta std::vector<@mtype> types { @dynamic_type(types_t)... };

  // Use an ordinary function to sort and unique types.
  @meta make_unique(types);

  // @pack_type returns an array/std::vector<@mtype> as a type parameter pack.
  // Print the unique list of names as a diagnostic.
  @meta std::cout<< @type_name(@pack_type(types))<< "\n"...;

  // Typedef a tuple_t over these unique types.
  typedef tuple_t<@pack_type(types)...> type_t;
};

int main() {
  // Create a tuple of just the unique types in the arguments list.
  // This has only four data members.
  typename unique_tuple_t<int, double, char*, double, char*, float>::type_t 
    tuple { };

  return 0;
}
```
```
$ circle unique.cxx
int
float
double
char*
```

## Introspection - serialize structs generically

[**serialization.cxx**](serialization.cxx)
```cpp
#include <iostream>
#include <string>

template<typename type_t>
concept C1 = std::is_class<type_t>::value;

template<C1 type_t>
void print_obj(const type_t& obj) {
  // Loop over each non-static data member.
  @meta for(int i = 0; i < @member_count(type_t); ++i) {
    // Print its member name and member value.
    std::cout<< @member_name(type_t, i)<< " : "<< @member_ref(obj, i)<< "\n";
  }
}

struct foo_t {
  int a;
  double b;
  const char* c;
  std::string d;
};

int main() {
  foo_t foo {
    5,
    3.14,
    "A C string",
    "A C++ string"
  };

  print_obj(foo);
  return 0;
}
```
```
$ circle serialization.cxx
$ ./serialization 
a : 5
b : 3.14
c : A C string
d : A C++ string
```

Feed compile-time loops with introspection expressions like `@member_count` to control the number of steps. At each step, use queries like `@member_name` (to get the name of a class member as a string literal) and `@member_ref` (to access the i'th data member) to serialize objects.

## Reflection - inject functions from the contents of a JSON

[**inject.json**](inject.json*)
```json
{
  "sq"     : "x * x",
  "unity"  : "sq(sin(x)) + sq(cos(x))"
}
```
[**inject.cxx**](inject.cxx)
```cpp
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
```
```
$ circle inject.cxx
Injecting sq = x * x
Injecting unity = sq(sin(x)) + sq(cos(x))
$ ./inject
unity(.3) = 1
```

## Organizing injection with Circle macros

[**inject2.cxx**](inject2.cxx)
```cpp
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

@macro void inject_from_json(std::string filename) {
  // Load a JSON file at compile time. These objects have automatic storage
  // duration at compile time. They'll destruct when the end of the macro
  // is hit.
  @meta std::ifstream inject_file(filename);
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
```

Circle macros allow you to modularize your metapogramming. They compose like normal functions, but when called, expand directly into the calling scope. You can even expand them into specific namespaces from any kind of scope.

