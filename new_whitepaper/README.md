# The Circle programming language

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

Put the `@meta` keyword in front of a statement to execute it at compile time. There's an integrated interpreter for executing function definitions, and you can make foreign function calls to routines that are defined externally. 

Compile-time control flow changes the execution of source translation, allowing types and functions to be defined in a data-dependent, imperative fashion.

## The design

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
    * Extensions to inject type-ids, expressions, statements and entire files.
* `@mtype` encapsulates a type-id in a variable.
    * `@dynamic_type` and `@static_type` convert between types and `@mtype`.
    * Store in standard containers, sort them, unique them. 
* Powerful macros.
    * Circle macros undergo argument deduction and overload resolution like normal functions.
    * Expand their contents into the calling scope.
    * Create a meta context to hold compile-time variables.

## Examples:

* [Dynamic names - turn strings and ints into identifiers](#dynamic-names---turn-strings-and-ints-into-identifiers)
* [Tuple - a flat data structure](#tuple---a-flat-data-structure)
* [Unique types - imperative metaprogramming](#unique-types---imperative-metaprogramming)
* [Introspection - serialize structs generically](#introspection---serialize-structs-generically)
* [Introspection on enums](#introspection-on-enums)
* [Matching enums with functions](#matching-enums-with-functions)
* [Reflection - inject functions from the contents of a JSON](#reflection---inject-functions-from-the-contents-of-a-json)
* [Organizing injection with Circle macros](#organizing-injection-with-circle-macros)
* [Typed enums - a first-class type list](#typed-enums---a-first-class-type-list)
* [Typed enums 2 - type list operations](#typed-enums-2---type-list-operations)
* [Typed enums 3 - join and unique](#typed-enums-3---join-and-unique)
* [A variant class](#a-variant-class)
* [Generating downcast visitors](#generating-downcast-visitors)
* [Expression macros for eprintf](#expression-macros-for-eprintf)
* [Embedded domain-specific languages](#embedded-domain-specific-languages)
* [A hand-rolled DSL: Reverse Polish Notation](#a-hand-rolled-dsl-reverse-polish-notation)

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

Circle adds a _for-enum-statement_, which conveniently iterates over the enumerators in an enum. Because this type information is only known at compile time, the statement must be prefixed with `@meta`. The loop index `e2` is not a constant (it's not even const), but it _is_ constexpr for the purpose of building _constant-expressions_. We use it emit a _case-statement_ for each enumerator.

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

Here we join two type lists and uniqueify their associated types. The _enum-specifier_ syntax has been extended to allow semicolon-separated statements. The `types` declaration holds `@mtype` objects of the associated types; this vector is transformed by `make_unique`, then expanded into a type parameter pack with `@pack_type`, which serves as the enum definition.

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

## Expression macros for eprintf

[**eprintf.cxx**](eprintf.cxx)
```cpp
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdio>

// Scan through until the closing '}'.
inline const char* parse_braces(const char* text) {
  const char* begin = text;

  while(char c = *text) {
    if('{' == c)
      return parse_braces(text + 1);
    else if('}' == c)
      return text + 1;
    else
      ++text;    
  }

  throw std::runtime_error("mismatched { } in parse_braces");
}

// Edit the format specifier. Dump the text inside 
inline void transform_format(const char* fmt, std::string& fmt2, 
  std::vector<std::string>& exprs) {

  std::vector<char> text;
  while(char c = *fmt) {
    if('{' == c) {
      // Parse the contents of the braces.
      const char* end = parse_braces(fmt + 1);
      exprs.push_back(std::string(fmt + 1, end - 1));
      fmt = end;
      text.push_back('%');
      text.push_back('s');

    } else if('%' == c && '{' == fmt[1]) {
      // %{ is the way to include a { character in the format string.
      fmt += 2;
      text.push_back('{');

    } else {
      ++fmt;
      text.push_back(c);
    }
  }

  fmt2 = std::string(text.begin(), text.end());
}

@macro auto eprintf(const char* fmt) {
  // Process the input specifier. Remove {name} and replace with %s.
  // Store the names in the array.
  @meta std::vector<std::string> exprs;
  @meta std::string fmt2;
  @meta transform_format(fmt, fmt2, exprs);

  // Convert each name to an expression and from that to a string.
  // Pass to sprintf via format.
  return printf(
    @string(fmt2.c_str()), 
    std::to_string(@expression(@pack_nontype(exprs))).c_str()...
  );
}


int main() {
  double x = 5;
  eprintf("x = {x} sqrt = {sqrt(x)} exp = {exp(x)}\n");

  return 0;
}
```
```
$ circle eprintf.cxx 
$ ./eprintf 
x = 5.000000 sqrt = 2.236068 exp = 148.413159
```

The `eprintf` is an unhygienic kind of printf. Rather than separately annotating an expression in the format specifier (with `%f` or `%s`, or perhaps with argument ordinals like `{0}`), we state the expression itself inside the format specifier:

```cpp
  double x = 5;
  eprintf("x = {x} sqrt = {sqrt(x)} exp = {exp(x)}\n");
```

The eprintf implementation will actually execute the three braced expressions from the scope in which it's called. This requires some real metaprogramming.

First, implement `eprintf` as an _expression macro_ rather than a normal function. We indicate this by having it return `auto`. (Statement macros return void.) The macro will be expanded from the scope of its call, which is critical, because `x` would not be found by name lookup otherwise.

Since Circle allows execution of arbitrary code at compile time, we implement a `transform_format` function that scans for the braces in the format specifier, moves the contents into a vector of strings, and replaces the braces by `%s` printf escapes. We then use `@expression` to inject each of these strings as expressions, and pass the result objects to `std::to_string` to turn into strings. This operation is done simultaneously for all escapes in the format specifier by way of a parameter pack expansion.

While eprintf may seem a bit gratuitous, consider the format specifier as a domain-specific language. Expression macros and `@expression` code injection are critical features for integrating DSLs with the surrounding C++ code.

## Embedded domain-specific languages

[**peg_dsl.cxx**](peg_dsl.cxx)
```cpp
// http://www.circle-lang.org/
// A Circle language example using an open source dynamic BNF parser to 
// implement a simple 

// The parser:
// https://github.com/yhirose/cpp-peglib

#include "peglib.h"
#include <cstdio>
#include <stdexcept>

// Define a simple grammar to do a 5-function integer calculation.
@meta peg::parser peg_parser(R"(
  EXPRESSION       <-  TERM (TERM_OPERATOR TERM)*
  TERM             <-  FACTOR (FACTOR_OPERATOR FACTOR)*
  FACTOR           <-  NUMBER / IDENTIFIER / '(' EXPRESSION ')'

  TERM_OPERATOR    <-  < [-+] >
  FACTOR_OPERATOR  <-  < [/*%] >
  NUMBER           <-  < [0-9]+ >
  IDENTIFIER       <-  < [_a-zA-Z] [_0-9a-zA-Z]* >

  %whitespace      <-  [ \t\r\n]*
)");

// peg-cpplib attaches semantic actions to each rule to construct an AST.
@meta peg_parser.enable_ast();

@macro auto peg_dsl_eval(const peg::Ast& ast) {
  @meta if(ast.name == "NUMBER") {
    // Put @meta at the start of an expression to force stol's evaluation
    // at compile time, which is when ast->token is available. This will turn
    // the token spelling into an integral constant at compile time.
    return (@meta stol(ast.token));

  } else @meta if(ast.name == "IDENTIFIER") {
    // Evaluate the identifier in the context of the calling scope.
    // This will find the function parameters x and y in dsl_function and
    // yield lvalues of them.
    return @expression(ast.token);

  } else {
    // We have a sequence of nodes that need to be folded. Because this is an
    // automatically-generated AST, we just have an array of nodes where 
    // the even nodes are FACTOR and the odd nodes are OPERATORs.
    // A bespoke AST would left-associate the array into a binary tree for
    // evaluation that more explicitly models precedence.

    @meta const auto& nodes = ast.nodes;
    return peg_dsl_fold(nodes.data(), nodes.size());
  }
}

template<typename node_t>
@macro auto peg_dsl_fold(const node_t* nodes, size_t count) {
  static_assert(1 & count, "expected odd number of nodes in peg_dsl_fold");

  // We want to left-associate a run of expressions.

  @meta if(1 == count) {
    // If we're at a terminal in the expression, evaluate the FACTOR and
    //  return it.
    return peg_dsl_eval(*nodes[0]);

  } else {
    // Keep descending until we're at a terminal. To left associate, fold
    // everything on the left with the element on the right. For the
    // expression
    //   a * b / c % d * e    this is evaluated as 
    //   (a * b / c % d) * e, 
    // where the part in () gets recursively processed by peg_dsl_fold.

    // Do a left-associative descent.

    // Since the DSL has operators with the same token spellings as C++,
    // we can just use @op to handle them all generically, instead of switching
    // over each token type.
    return @op(
      nodes[count - 2]->token,
      peg_dsl_fold(nodes, count - 2),
      peg_dsl_eval(*nodes[count - 1])
    );
  }
}

@macro auto peg_dsl_eval(const char* text) {
  @meta std::shared_ptr<peg::Ast> ast;
  @meta if(peg_parser.parse(text, ast)) {
    // Generate code for the returned AST as an inline expression in the
    // calling expression.
    return peg_dsl_eval(*ast);

  } else
    @meta throw std::runtime_error("syntax error in PEG expression")
}

////////////////////////////////////////////////////////////////////////////////

long dsl_function(long x, long y) {
  // This function has a DSL implementation. The DSL text is parsed and 
  // lowered to code when dsl_function is translated. By the time it is called,
  // any remnant of peg-cpplib is gone, and only the LLVM IR or AST remains.
  return x * peg_dsl_eval("5 * x + 3 * y + 2 * x * (y % 3)") + y;
}

int main(int argc, char** argv) {
  if(3 != argc) {
    printf("Usage: peg_dsl [x] [y]\n");
    exit(1);
  }

  int x = atoi(argv[1]);
  int y = atoi(argv[2]);

  int z = dsl_function(x, y);
  printf("result is %d\n", z);

  return 0;
}
```
```
$ circle peg_dsl.cxx
$ ./peg_dsl 5 10
result is 335
```

This wild program builds off the [eprintf.cxx](eprintf.cxx) example. We start by including an [open-source PEG parser](https://github.com/yhirose/cpp-peglib). This header-only parser doesn't know anything about Circle. It's intended for runtime execution. But since Circle hash an integrated interpreter, we'll use it at compile time to parse our own domain-specific language into an AST, then traverse the AST with Circle macros and lower the input to C++ code. Effectively we use C++-as-script to implement a new language frontend, and target C++ as a backend.

The DSL is just a simple five-function infix calculator. The cpp-peglib parser came with a similar grammar as an example, so we took that and modified it slightly, to support IDENTIFIERs. Two rounds of parsing are needed to implement the DSL: First, the grammar as BNF is parsed by cpp-peglib into some internal data structures; Second, the input text in `dsl_function` is through cpp-peglib, and this time those data structures are applied to transform the text into an AST that conforms to the specified grammar.

Expression macros are used to lower the parser's AST to C++. This kind of macro is very limited--its only non-meta statement must be a single _return-statement_. The argument for the return is essentially broken off and substituted for the macro expansion at the point where it's called. Since we are traversing an AST and returning subexpressions to represent each node, we are translating the DSL into C++ code as a single complex expression. The _unoptimized_ code when executing `peg_dsl_eval` is exactly the code embedded inside the quotes. All the recursive macro expansion work yields a single subexpression. We're able to generate exactly the code we intended, and don't rely on the optimizer to inline out many layers of function calls, like template metaprogramming DSLs do.

## A hand-rolled DSL: Reverse Polish Notation

[**rpn.cxx**](rpn.cxx)
```cpp
#include <cmath>
#include <memory>
#include <iostream>
#include <sstream>
#include <stack>

namespace rpn {

enum class kind_t {
  var,       // A variable or number
  op,        // + - * /
  f1,        // unary function
  f2,        // binary function
};

struct token_t {
  kind_t kind;
  std::string text;
};

struct token_name_t {
  kind_t kind;
  const char* string;
};

const token_name_t token_names[] {
  // Supported with @op
  { kind_t::op,    "+"     },
  { kind_t::op,    "-"     },
  { kind_t::op,    "*"     },
  { kind_t::op,    "/"     },

  // Unary functions
  { kind_t::f1,    "abs"   },
  { kind_t::f1,    "exp"   },
  { kind_t::f1,    "log"   },
  { kind_t::f1,    "sqrt"  },
  { kind_t::f1,    "sin"   },
  { kind_t::f1,    "cos"   },
  { kind_t::f1,    "tan"   },
  { kind_t::f1,    "asin"  },
  { kind_t::f1,    "acos"  },
  { kind_t::f1,    "atan"  },

  // Binary functions
  { kind_t::f2,    "atan2" },
  { kind_t::f2,    "pow"   },
};

inline kind_t find_token_kind(const char* text) {
  for(token_name_t name : token_names) {
    if(!strcmp(text, name.string))
      return name.kind;
  }
  return kind_t::var;
}

struct node_t;
typedef std::unique_ptr<node_t> node_ptr_t;

struct node_t {
  kind_t kind;
  std::string text;
  node_ptr_t a, b;
};

inline node_ptr_t parse(const char* text) {
  std::istringstream iss(text);
  std::string token;

  std::stack<node_ptr_t> stack;
  while(iss>> token) {
    // Make operator ^ call pow.
    if("^" == token)
      token = "pow";

    // Match against any of the known functions.
    kind_t kind = find_token_kind(token.c_str());

    node_ptr_t node = std::make_unique<node_t>();
    node->kind = kind;
    node->text = token;
    
    switch(kind) {
      case kind_t::var:
        // Do nothing before pushing the node.
        break;      

      case kind_t::f1:
        if(stack.size() < 1)
          throw std::range_error("RPN formula is invalid");

        node->a = std::move(stack.top()); stack.pop();
        break;

      case kind_t::op:
      case kind_t::f2:
        // Binary operators and functions pop two items from the stack,
        // combine them, and push the result.
        if(stack.size() < 2)
          throw std::range_error("RPN formula is invalid");

        node->b = std::move(stack.top()); stack.pop();
        node->a = std::move(stack.top()); stack.pop();
        break;
    }

    stack.push(std::move(node));
  }

  if(1 != stack.size())
    throw std::range_error("RPN formula is invalid");

  return std::move(stack.top());
}

@macro auto eval_node(const node_t* __node) {
  @meta+ if(rpn::kind_t::var == __node->kind) {
    @emit return @expression(__node->text);

  } else if(rpn::kind_t::op == __node->kind) {
    @emit return @op(
      __node->text, 
      rpn::eval_node(__node->a.get()),
      rpn::eval_node(__node->b.get())
    );

  } else if(rpn::kind_t::f1 == __node->kind) {
    // Call a unary function.
    @emit return @(__node->text)(
      rpn::eval_node(__node->a.get())
    );

  } else if(rpn::kind_t::f2 == __node->kind) {
    // Call a binary function.
    @emit return @(__node->text)(
      rpn::eval_node(__node->a.get()),
      rpn::eval_node(__node->b.get())
    );
  }
}

// Define an expression macro for evaluating the RPN expression. This
// expands into the expression context from which it is called. It can have
// any number of meta statements, but only one real return statement, and no
// real declarations (because such declarations are prohibited inside
// expressions).
@macro auto eval(const char* __text) {
  @meta rpn::node_ptr_t __node = rpn::parse(__text);
  return rpn::eval_node(__node.get());
}

} // namespace rpn

double my_function(double x, double y, double z) {
  return rpn::eval("z 1 x / sin y * ^");
}

int main() {
  double result = my_function(.3, .6, .9);
  printf("%f\n", result);
  return 0;
}
```

Here we write a new DSL by hand: Reverse Polish Notation. RPN doesn't really even have a syntax, so using a BNF parser like cpp-peglib is of limited value. Instead, we'll use `std::istringstream` to read in a token at a time, then take appropriate action.

In this case, appropriate action is transforming RPN into infix notation using a stack. We don't evaluate the expression as we go, as an interpreter might do, because our intent is to use RPN to generate code. The result of the RPN parsing is a hierarchical AST, which we can lower to C++ code using expression macros.

To make our DSL almost as powerful as a 1970s HP handheld calculator, we flag a number of unary and binary elementary functions in addition to the four basic arithmetic operations. The spelling of these functions are saved in the AST nodes. 

When lowering the AST to C++ code, the function names (which are known at compile-time, since we parsed the RPN at compile time) are run through the dynamic name operator `@()`, which yields identifiers for each of them. When used in expressions, this triggers ordinary C++ name lookup, allowing each function name to find its corresponding function lvalue from `<cmath>`.

By first transforming the RPN text into an AST, we allow evaluation to proceed without involving a stack. Expression macros have their _return-statement_ arguments detached and substituted at the point of call, producing one complex expression in `my_function`. 

After parsing to AST and lowering AST to C++, the RPN input `z 1 x / sin y * ^`, which has precedences `(z (((1 x /) sin) y *) ^)`, is transformed into an expression that is absolutely equivalent to `pow(z, sin(1 / x) * y)`. 

Circle allows extending the language with DSLs. The frontend is written with compile-time C++ (think C++ as a script language) and the backend targets the C++ frontend.

