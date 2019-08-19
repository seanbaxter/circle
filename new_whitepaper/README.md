# Circle
The C++ Automation Language  
2019  
Sean Baxter

> _"it's like c++ that went away to train with the league of shadows and came back 15 years later and became batman"_

Download [here](https://www.circle-lang.org/)

Follow me on Twitter [@seanbax](https://www.twitter.com/seanbax) for compiler updates.

* Start with C++.
* Integrate an interpreter. 
    * Anything can be run at compile time.
    * Full access to the host environment.
* Reflection.
    * Simply interleave compile-time control flow with regular declarations.
* Introspection keywords.
    * No runtime cost. Introspection gives access to info already maintained by the compiler.
* Dynamic names.
    * `@()` turns strings and integers into identifiers.
* Rich parameter packs.
    * Parameter packs now separated from templates.
    * Many new extensions return parameter packs.
    * Subscript template and function parameter packs with `...[]`.
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

Put the `@meta` keyword in front of a statement to execute it at compile time. There's an integrated interpreter for executing function bodies, and you can make foreign function calls to routines that are defined externally.

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

Loop over each element in a parameter pack and declare a non-static data member for it. The type is `types_t...[i]`, which is the i'th element of the pack. The name is `@(i)`, which is an identifier like `_0`, `_1`, and so on.

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
void stable_unique(std::vector<type_t>& vec) {
  // std::sort + std::unique also works, but isn't stable.
  std::vector<type_t> vec2;
  for(type_t x : vec) {
    if(vec2.end() == std::find(vec2.begin(), vec2.end(), x))
      vec2.push_back(x);
  }
  vec = std::move(vec2);
}

template<typename... types_t>
struct unique_tuple_t {
  // Create a compile-time std::vector<@mtype>. @mtype encapsulates a 
  // type, allowing you to manipulate it like a variable. This means we can
  // sort them! @dynamic_type converts a type to an @mtype prvalue.
  @meta std::vector<@mtype> types { @dynamic_type(types_t)... };

  // Use an ordinary function to unique these types.
  @meta stable_unique(types);

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
double
char*
float
```

`unique_tuple_t` typedefs a `tuple_t` over a unique list of its template arguments. Unlike with Standard C++, Circle does this with no template metaprogramming abuse:
1. Convert the argument types into `@mtype` variables and load into a vector. `@mtype` is a pointer-sized builtin type that encapsulates a cv-qualified type.
1. Sort and unique the vector using STL algorithms.
1. Expand the types into a `tuple_t` argument list with `@pack_type`--this yields a type parameter pack by extracting the types out of their `@mtype` objects in an array or vector.

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

## Introspection on enums

```cpp
#include <iostream>
#include <type_traits>

template<typename type_t>
const char* name_from_enum(type_t e) {
  static_assert(std::is_enum<type_t>::value);

  switch(e) {
    // A ranged-for over the enumerators in this enum.
    @meta for enum(type_t e2 : type_t) {
      // e2 is known at compile time, so we can use it in a case-statement.
      case e2:
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

int main() {

  enum color_t {
    red, blue, green, yellow, purple, violet, chartreuse, puce,
  };
  color_t colors[] { yellow, violet, puce };

  // Print all the color names in the array.
  for(color_t c : colors)
    std::cout<< name_from_enum(c)<< "\n";

  return 0;
}
```
```
$ circle enums.cxx
$ ./enums
yellow
violet
puce
```

Circle adds a _for-enum_statement_, which conveniently iterates over the enumerators in an enum. Because this type information is only known at compile time, the statement must be prefixed with `@meta`. The loop index `e2` is not a constant (it's not even const), but it _is_ constexpr for the purpose of building _constant-expressions_. We use it emit a _case-statement_ for each enumerator.

Without introducing a runtime type info, we're able to build a generic function that maps an enumerator value to the enumerator name. 

## Matching enums with functions

[**shapes.cxx**](shapes.cxx)
```cpp
#include "util.hxx"

enum class shape_t {
  circle,
  triangle,
  square, 
  hexagon, 
  octagon,
};

double circle(double r) {
  return M_PI * r * r;
}

double triangle(double r) {
  return r * sqrt(3) / 2;
}

double square(double r) {
  return 4 * r * r;
}

double hexagon(double r) {
  return 1.5 * sqrt(3) * (r * r);
}

double octagon(double r) {
  return 2 * sqrt(2) * (r * r);
}

double polygon_area(shape_t shape, double r) {
  switch(shape) {
    @meta for enum(shape_t s : shape_t) {
      case s: 
        // Call the function that has the same name as the enumerator.
        return @(@enum_name(s))(r);
    }

    default: 
      assert(false);
      return 0;
  }
}

template<typename type_t>
std::string enum_to_string(type_t x) {
  static_assert(std::is_enum<type_t>::value);

  switch(x) {
    @meta for enum(auto y : type_t)
      case y: return @enum_name(y);

    default: return "<" + std::to_string((int)x) + ">";
  }
}

template<typename type_t>
type_t string_to_enum(const char* name) {
  static_assert(std::is_enum<type_t>::value);

  @meta for enum(type_t x : type_t) {
    if(!strcmp(name, @enum_name(x)))
      return x;
  }

  throw std::runtime_error(format("%s is not an enumerator of %s", 
    name, @type_name(type_t)).c_str());
}

const char* usage = "  shapes <shape-name> <radius>\n";

int main(int argc, char** argv) {
  if(3 != argc) {
    puts(usage);
    exit(1);
  }

  shape_t shape = string_to_enum<shape_t>(argv[1]);
  double radius = atof(argv[2]);

  double area = polygon_area(shape, radius);

  printf("Area of %s of radius %f is %f\n", enum_to_string(shape).c_str(),
    radius, area);

  return 0;
}
```
```
$ circle shapes.cxx
$ ./shapes hexagon 3
Area of hexagon of radius 3.000000 is 23.382686
$ ./shapes octagon 3
Area of octagon of radius 3.000000 is 25.455844
$ ./shapes trapezoid 3
terminate called after throwing an instance of 'std::runtime_error'
  what():  trapezoid is not an enumerator of shape_t
Aborted (core dumped)
```
The dynamic name operator `@()` lets us associate different kinds of declarations that have the same name. The five enumerators in the scoped enum `shape_t` have corresponding functions that, given a radius, return their areas. In Standard C++, this association cannot be exploited. But with Circle, we can use the introspection expression `@enum_name` to yield an enumerator name, feed it to the dynamic name operator `@()` to create an identifier, and perform ordinary name lookup to match the area functions.

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

Circle's interpreter gives unrestricted access to the host environment. We use iostreams and a JSON parsing library to load a JSON file at compile time. Compile-time control flow lets us traverse the JSON tables. Reflection lets us declare a C++ function for each JSON item. The `@expression` keyword converts the contained text to C++ code, which is returned as part of the function definition.

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

## Typed enums - a first-class type list

[**typed_enums.cxx**](typed_enums.cxx)
```cpp
#include <iostream>

// "enum typename" starts a typed enum. This includes an optional identifier
// and a mandatory type-id. Use @enum_type to extract the associated type
// of a typed enum.
enum typename type_list_t {
  int,
  double*, 
  char*, 
  void, 
  float[4]
};

int main() {
  // Walk through the enum and print the associated types.
  std::cout<< "type_list_t with a for-enum loop:\n";
  @meta for enum(auto e : type_list_t)
    std::cout<< @type_name(@enum_type(e))<< "\n"; 
  std::cout<< "\n";

  // We can do the same thing with an parameter pack.
  std::cout<< "type_list_t with a pack expansion:\n";
  std::cout<< @type_name(@enum_types(type_list_t))<< "\n" ...;

  return 0;
}
```
```
$ circle typed_enums.cxx
$ ./typed_enums
type_list_t with a for-enum loop:
int
double*
char*
void
float[4]

type_list_t with a pack expansion:
int
double*
char*
void
float[4]
```

Circle adds a twist on the enumeration, the _typed enum_. These behave like normal enums, but each enumerator has an associated type. This association is maintained by the compiler, and can be accessed with the `@enum_type` and `@enum_types` extensions. If an enum name is not provided, it is assigned a name like `_0`, `_1` and so on.

## Typed enums 2 - type list operations

[**typed_enums2.cxx**](typed_enums2.cxx)
```cpp
#include <iostream>

enum typename type_list1_t {
  // Circle enums may be specified with semicolon-separated statements.
  int;
  double*;
  char*;
};

enum typename type_list2_t {
  // or comma-separation declarations.
  void**,
  int[3],
  char32_t
};

enum typename joined_t {
  // We can programmatically inject with compile-time control flow
  @meta for enum(auto e : type_list1_t)
    @enum_type(e);

  // Or with parameter pack trickery.
  @enum_types(type_list2_t)...;
};

// Print all the associated types in joined_t
@meta std::cout<< @type_name(@enum_types(joined_t))<< "\n" ...;

int main() {
  return 0;
}
```
```
$ circle typed_enums2.cxx
int
double*
char*
void**
int[3]
char32_t
```

Circle supports _enum-specifiers_ definitions with both comma-separated and semicolon-separated declarations. This lets us generate enums with compile-time control flow. `joined_t` is a type list that joins `type_list1_t` with `type_list2_t`. 

## Typed enums 3 - join and unique

[**typed_enums3.cxx**](typed_enums3.cxx)
```cpp
#include <iostream>
#include <algorithm>
#include <vector>

template<typename type_t>
void stable_unique(std::vector<type_t>& vec) {
  std::vector<type_t> vec2;
  for(type_t x : vec) {
    if(vec2.end() == std::find(vec2.begin(), vec2.end(), x))
      vec2.push_back(x);
  }
  vec = std::move(vec2);
}

enum typename type_list1_t {
  int,
  double*,
  char*,
  char32_t,
};

enum typename type_list2_t {
  void**,
  int[3],
  double*,
  int[3],
  char32_t
};

enum typename uniqued_t {
  // Make a std::vector<@mtype> holding all associated types from 
  // type_list1_t and type_list2_t.
  // The types array has automatic storage duration, determined by the
  // braces of the enum-specifier.
  @meta std::vector<@mtype> types {
    @dynamic_type(@enum_types(type_list1_t))...,
    @dynamic_type(@enum_types(type_list2_t))...
  };

  // Use STL algorithms to unique this vector.
  @meta stable_unique(types);

  // Declare a typed enum with these associated types.
  @pack_type(types)...;
};

// Print all the associated types in uniqued_t
@meta std::cout<< @type_name(@enum_types(uniqued_t))<< "\n" ...;

int main() {
  return 0;
}
```
```
$ circle typed_enums3.cxx
int
double*
char*
char32_t
void**
int[3]
```

Here we join two type lists and uniqueify their associated types. The _enum-specifier_ syntax has been extended to allow semicolon-separated statements. The `types` declaration holds `@mtype` objects of the associated types; this vector is transformed by `make_unique`, then expanded into a type parameter pack with `@pack_type`, which serevs as the enum definition.

## A variant class

[**variant.cxx**](variant.cxx)
```cpp
#include <iostream>

template<typename type_list>
struct variant_t {
  static_assert(__is_typed_enum(type_list));

  // Create an instance of the enum.
  type_list kind { };

  union {
    // Create a variant member for each enumerator in the type list.
    @meta for enum(auto e : type_list)
      @enum_type(e) @(@enum_name(e));
  };

  // For a real variant, implement the default and copy/move ctors, assignment
  // operators, etc. These use similar for-enum loops to perform actions on the
  // active variant member.

  // Implement a visitor. This calls the callback function and passes the
  // active variant member. f needs to be a generic lambda or function object
  // with a templated operator().
  template<typename func_t>
  auto visit(func_t f) {
    switch(kind) {
      @meta for enum(auto e : type_list) {
        case e:
          return f(@(@enum_name(e)));
          break;      
      }
    }
  }
};

// Define a type list to be used with the variant. Each enumerator identifier
// maps to a variant member name. Each associated type maps to a variant
// member type.
enum typename class my_types_t {
  i = int, 
  d = double,
  s = const char* 
};

int main() {
  auto print_arg = [](auto x) {
    std::cout<< @type_name(decltype(x))<< " : "<< x<< "\n";
  };

  variant_t<my_types_t> var;

  // Fill with a double. var.d is the double variant member.
  var.kind = my_types_t::d;
  var.d = 3.14;
  var.visit(print_arg);

  // Fill with a string. var.s is the const char* variant member.
  var.kind = my_types_t::s;
  var.s = "Hello variant";
  var.visit(print_arg);

  return 0;
}
```
```
$ circle variant.cxx 
$ ./variant
double : 3.14
const char* : Hello variant
```

The Circle variant leverages the power of typed enums. The `my_types_t` typed enum serves as a manifest for a variant specialization. An enum data member indicates the active variant member. The variant definition then loops over each enumerator and declares a variant member with the name of the enumerator and with its associated type. We specify enumerator names `i`, `d` and `s`, and we have these member names in the variant specialization.

## Generating downcast visitors

[**enum_dispatch.cxx**](enum_dispatch.cxx)
```cpp
#include <type_traits>
#include <cassert>
#include <iostream>

struct ast_literal_t;
struct ast_string_t;
struct ast_unary_t;
struct ast_binary_t;
struct ast_call_t;

struct ast_t {
  // Associate the base class's enums with each concrete type.
  enum typename kind_t {
    kind_literal = ast_literal_t,
    kind_string  = ast_string_t,
    kind_unary   = ast_unary_t,
    kind_binary  = ast_binary_t,
    kind_call    = ast_call_t,
  } kind;

  ast_t(kind_t kind) : kind(kind) { }

  template<typename type_t>
  bool isa() const {
    // Find the enumerator in kind_t that has the associated type type_t.
    static_assert(std::is_base_of<ast_t, type_t>::value);
    return @type_enum(kind_t, type_t) == kind;
  }

  // Perform an unconditional downcast from ast_t to a derived type.
  // This is like llvm::cast.
  template<typename type_t>
  type_t* cast() {
    assert(isa<type_t>());
    return static_cast<type_t*>(this);
  }

  // Perform a conditional downcast. This is like llvm::dyn_cast.
  template<typename type_t>
  type_t* dyn_cast() {
    return isa<type_t>() ? cast<type_t>() : nullptr;
  }
};

struct ast_literal_t : ast_t {
  ast_literal_t() : ast_t(kind_literal) { }
};

struct ast_string_t : ast_t {
  ast_string_t() : ast_t(kind_string) { }
};

struct ast_unary_t : ast_t {
  ast_unary_t() : ast_t(kind_unary) { }
};

struct ast_binary_t : ast_t {
  ast_binary_t() : ast_t(kind_binary) { }
};

struct ast_call_t : ast_t {
  ast_call_t() : ast_t(kind_call) { }
};

// Generate code to visit each concrete type from this base type.
void visit_ast(ast_t* ast) {

  template<typename type_t>
  @macro void forward_declare() {
    // Forward declare visit_ast on this type.
    void visit_ast(type_t* derived);
  }

  switch(ast->kind) {
    @meta for enum(auto e : ast_t::kind_t) {
      case e: 
        // Forward declare a function on the derived type in the global ns.
        @macro namespace() forward_declare<@enum_type(e)>();

        // Downcast the AST pointer and call the just-declared function.
        return visit_ast(ast->cast<@enum_type(e)>());
    }
  }
}

// Automatically generate handlers for each concrete type.
// In a real application, these are hand-coded and perform 
// type-specific actions.
@meta for enum(auto e : ast_t::kind_t) {
  void visit_ast(@enum_type(e)* type) {
    std::cout<< "visit_ast("<< @type_name(@enum_type(e))<< "*) called\n";
  }
}

int main() {
  ast_binary_t binary;
  visit_ast(&binary);

  ast_call_t call;
  visit_ast(&call);

  return 0;
}
```
```
$ circle enum_dispatch.cxx 
$ ./enum_dispatch 
visit_ast(ast_binary_t*) called
visit_ast(ast_call_t*) called
```

Circle reduces boilerplate involved with class inheritance. A typed enum is used to associate all concrete types in the base type (`ast_t`) of an inheritance family with the `kind` member. We can then perform conditional downcasts from the base type to any of the concrete types using introspection on this type list. The [LLVM dyn_cast](http://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates) pattern requires a `classof` declaration in each derived type, but we put this functionality in the `isa` function in the base class.

A visitor pattern that examines the `kind` member of a base, performs the correct downcast, and dispatches to the derived type's function is implemented algorithmically, with a _for-enum_ over the enumerators in the base's type list. This kind of metaprogramming can be pushed much further, allowing parametric base types, return types, parameters and names for the visitor function.

