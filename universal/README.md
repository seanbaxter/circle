# Universal Member Access

This capability grew out of the [list comprehension](../comprehension/comprehenion.md) and [pattern matching](../pattern/pattern.md) Circle extensions. Those sections give a more expensive view into the mechanisms detailed here.

## Contents.

1. [Structured bindings](#structured-bindings)
2. [Pack subscripts and slices](#pack-subscripts-and-slices)
3. [Tuple subscripts and slices](#tuple-subscripts-and-slices)
4. [Implicit slices](#implicit-slices)
5. [Object lengths](#object-lengths)
6. [Pack indices](#pack-indices)
7. [Transforming objects](#transforming-objects)

## Structured bindings.

C++17 introduced _structured bindings_, a way to disaggregate tuples, arrays and class objects into separate declarations.

[[**structured.cxx**]](structured.cxx)
```cpp
#include <tuple>
#include <iostream>

int main() {
  // Declare a tuple-like object.
  std::tuple<double, int, const char*> tuple(3.14, 100, "Hello tuple");

  // Bind temporaries to its parts.
  auto [a, b, c] = tuple;

  // Print its components.
  std::cout<< "0: "<< a<< "\n";
  std::cout<< "1: "<< b<< "\n";
  std::cout<< "2: "<< c<< "\n";
}
```
```
$ circle structured.cxx
$ ./structured
0: 3.14
1: 100
2: Hello tuple
```

Here, automatic storage duration objects _a_, _b_ and _c_ are declared and bound to the parts of the tuple. This is a _destructure_ operation. The C++ frontend first looks for specializations of `std::tuple_size<E>` on the operand type. In this case it finds it, so uses `std::tuple_element<I, E>` to access the type of each tuple part and `std::get<I>` to access each member lvalue.

[[**structured2.cxx**]](structured2.cxx)
```cpp
#include <tuple>
#include <iostream>

int main() {
  // Declare a tuple-like object.
  std::tuple<double, int, const char*> tuple(3.14, 100, "Hello tuple");

  // Bind temporaries to its parts.
  auto [...parts] = tuple;

  // Print its components.
  std::cout<< sizeof. tuple<< " components:\n";
  std::cout<< int...<< ": "<< parts<< "\n" ...;
}
```
```
$ circle structured2.cxx
$ ./structured2
3 components:
0: 3.14
1: 100
2: Hello tuple
```

Circle extends C++ by supporting structured binding onto parameter packs. Here _parts_ is a pack of automatic storage duration objects. Each pack element is bound to a tuple part. There are two big advantages here:
1. It's generic, because we don't have to know how many elements there are, and don't have to come up with separate names for them.
2. It provides access through pack expansion expressions, so we can print or mutate the parts, generically, in a single statement.

The pack structured binding is a foundation for this slick one-liner print, that works for tuples, pairs, std::array, builtin arrays, matrices and vectors (both first-class structured binding types in Circle), and all other non-union class objects.

This sample prompts an important question: why can't we generically print the tuple without declaring the structured binding? We only care about the structured binding as a means for exposing the tuple's elements as a non-type parameter pack in order to expand it in a pack-expansion expression. If we were to take the structured binding mechanisms and hoist them from declarations to expressions, we could perform disaggregation into parameter packs directly inside expressions.

## Pack subscripts and slices.

Circle includes the `...[subscript]` postfix operator to subscript parameter packs. It also includes the `...[begin:end:step]` slice operator to slice parameter packs. 

* `a...[subscript]` - yield the _subscript'th_ member of the pack _a_. This may be a type pack, non-type pack, template pack, or universal parameter pack.
* `a...[begin:end:step]` - reorder the elements of a parameter pack according to [extended slice rules](../comprehension#static-slices-on-template-parameter-packs).

## Tuple subscripts and slices.

We're going to add corresponding operators to support tuple-like operands. These new operators perform structured binding _as an expression_ and yield either a single element or a slice of the elements of the operand.

* `a.[subscript]` - yield the _subscript'th_ member of the object _a_.
* `a.[begin:end:step]` - yield a static parameter pack that [slices](../comprehension#static-slices-on-template-parameter-packs) the expression _a_.
* `sizeof.(a)` - return the number of tuple elements on public non-static data members in _a_, where _a_ is a type or _unary-expression_.
* `__is_structured_type(type)` - a trait indicating that the type is compatible with destructurization.

As with structured bindings, the subscript and slice operators first check for a specialization of `std::tuple_size<E>` before accessing data members of class types.

[**subscript.cxx**](subscript.cxx)
```cpp
#include <utility>
#include <tuple>
#include <array>
#include <iostream>

typedef float __attribute__((vector_size(16))) vec4;

template<typename type_t>
void print_object1(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  @meta for(int i = 0; i < sizeof...(type_t); ++i)
    std::cout<< "  "<< i<< ": "<< obj...[i]<< "\n";
}

template<typename type_t>
void print_object2(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int...<< ": "<< obj...[0:-1:1]<< "\n" ...;
}

template<typename type_t>
void print_object3(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";

  // Write comma-separated members inside braces.
  std::cout<< "  { "<< obj...[0];
  std::cout<< ", "<< obj...[1:]...;
  std::cout<< " }\n";
}

template<typename type_t>
void print_object4(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  if constexpr(__is_structured_type(type_t)) {
    std::cout<< "  { "<< obj...[0];
    std::cout<< ", "<< obj...[1:]...;
    std::cout<< " }\n";
  } else
    std::cout<< "  "<< obj<< "\n";
}

int main() {
  // std::pair is tuple-like.
  print_object1(std::make_pair(1, "Hello pair"));

  // std::tuple is tuple-like.
  print_object2(std::make_tuple(2, 1.618, "Hello tuple"));

  // std::array is tuple-like.
  print_object3(std::array { 5, 10, 15 } );

  // builtin arrays are structured binding types.
  int array[] { 20, 25, 30 };
  print_object4(array);

  // builtin vectors are structured binding types.
  print_object4(vec4(10, 20, 30, 40));

  // print a scalar. Rely on the __is_structure_type trait to choose the
  // right behavior.
  print_object4(100);
}
```
```
$ circle subscript.cxx && ./subscript 
std::pair<int, const char*>
  0: 1
  1: Hello pair
std::tuple<int, double, const char*>
  0: 2
  1: 1.618
  2: Hello tuple
std::array<int, 3>
  { 5, 10, 15 }
int[3]
  { 20, 25, 30 }
<4 x float>
  { 10, 20, 30, 40 }
int
  100
```

This sample generically prints the members of a pair, tuple, std::array, array and vector type. 

1. `print_object1` executes a compile-time loop over the members, ranging from 0 to `sizeof.(type_t)`. It uses `.[subscript]` to access each data member. 
2. `print_object2` reduces this to a one-liner. Use `.[begin:end:step]` slice notation to turn an object into a parameter pack of its members. The simple syntax `.[:]` is equivalent to `.[0:-1:1]`, meaning it ranges over all elements from left-to-right, stepping one at a time. The `int...` expression yields the current element of the pack expansion, which corresponds to the step count `i` in `print_object1`.
3. `print_object3` formats the elements into a comma-separated list enclosed in braces. The first element is written with a subscript. All subsequent elements are written, comma-prefixed, with a slice expansion. The `.[1:]` slice operation returns a pack of members starting at 1 and continuing to the end of the container.
4. `print_object4` adds a compile-time check, that tests if the object is a structured binding type. `__is_structured_type` returns true for tuple-like types (those with `std::tuple_size` specializations), arrays, matrices, vectors and non-union classes. This more generic function prints scalar types without requiring an overload.

## Implicit slices.

Static slices are a powerful mechanism, but look syntactically busy at times.

```cpp
auto tuple = make_tuple('a', 2, 3.0);
func(tuple...[:]...);
```

In contexts like this, you can expand the argument object directly, without explicitly slicing it.

```cpp
auto tuple = make_tuple('a', 2, 3.0);
func(tuple...);
```

You can expand an object operand in these contexts:
* Function argument list
* Template argument list
* Braced initializer list
* Unary fold expression

[**implicit.cxx**](implicit.cxx)
```cpp
#include <iostream>
#include <functional>
#include <tuple>
#include <array>

void func(auto... args) {
  std::cout<< args<< " "...;
  std::cout<< "\n";
}

template<auto... args>
struct foo_t { 
  foo_t() {
    std::cout<< @type_string(foo_t)<< "\n";
  }
};

int main() {
  // Expand array into a function argument list.
  int data1[] { 1, 2, 3 };
  func(0, data1..., 4);

  // Expand a normal array into an std::array.
  // Expand std::array into a function argument list.
  std::array data2 { data1..., 4, 5, 6 };
  func(data2..., 7);

  // Expand a tuple into a funtion argument list.
  auto tuple = std::make_tuple('a', 2u, 300ll);
  func(tuple...);

  // Use in a unary fold expression.
  int max = (... std::max data1);
  std::cout<< "max = "<< max<< "\n";

  int product = (... * data2);
  std::cout<< "product = "<< product<< "\n";

  // Specialize a template over compile-time data.
  // It can be constexpr.
  constexpr int data[] { 10, 20, 30 };
  foo_t<data...> obj1;

  // Or it can be meta.
  struct bar_t {
    int a;
    long b;
    char32_t c;
  };
  @meta bar_t bar { 100, 200, U'A' };

  // meta objects are mutable.
  @meta bar.b++;

  foo_t<bar...> obj2;
}
```

To be implicitly promoted to a static slice, the expression must be an object or parameter, of a tuple-like class, array, matrix, vector or any non-union class object. Universal member access implicitly splits it into its parts and inserts these into the function argument ist, template argument list, initializer list or unary fold expression.

Note that objects must be constexpr or meta to be valid template arguments.

## Object lengths.

[**length.cxx**](length.cxx)
```cpp
#include <tuple>
#include <utility>
#include <cstdio>

int main() {
  // sizeof... gives the length of a pack structured binding.
  auto [...pack] = std::make_pair(1, 2.0);
  printf("pack.length = %d\n", sizeof... pack);

  // sizeof... gives the number of members in a tuple-like type.
  printf("tuple.length = %d\n", sizeof... std::make_tuple('a', 2, 3.0));

  // sizeof... gives the length of the array.
  int my_array[] { 1, 2, 3, 4, 5, 6 };
  printf("my_array.length = %d\n", sizeof... my_array);

  // sizeof... gives the number of non-static public data members.
  struct obj_t {
    int x, y, z;
    const char* w;
  };
  printf("obj.length = %d\n", sizeof...(obj_t));
}
```
```
$ circle length.cxx && ./length
pack.length = 2
tuple.length = 3
my_array.length = 6
obj.length = 4
```

In Standard C++, `sizeof...(identifier)` yields the element count when pointed at a template parameter pack. The new `sizeof.` operator has an extended syntax and works on tuple-like entities:
* `sizeof. unary-expression` - Match a unary-expression.
* `sizeof.(type-id)` - Match a type-id.

Dependeng on its operand, `sizeof.` returns:
* Elements in a pack structured binding.
* Elements in a tuple-like type.
* Length of an array.
* Number of columns in a matrix.
* Number of components in a vector.
* Number of public non-static data members in a non-union class.

`sizeof.` is the long-awaited _ARRAY LENGTH OPERATOR_. 

## Pack indices.

C++ metaprogramming relies on the generation of integer sequences. The standard library class templates [`std::integer_sequence`](https://en.cppreference.com/w/cpp/utility/integer_sequence) and [`std::index_sequence`](https://en.cppreference.com/w/cpp/utility/index_sequence) allow argument deduction of integer non-type parameter packs, from which a function template can access and expand the deduced template parameters. Unfortunately, this is a burdensome way to use integer packs, as they must be deduced by a template, and cannot be used in line.

All mainline compiler frontends implement intrinsics to improve the performance of `std::integer_sequence`. For gcc/clang, this is [`__integer_pack`](https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/Type-Traits.html). In the mainline compilers, the intrinsics can't be expanded directly, and only serve as a compile-time optimization for the implementation of `std::integer_sequence`. In Circle, you can expand `__integer_pack`, or any other pack-yielding expression, directly from any expansion locus:

**integer_pack.cxx**
```cpp
template<int... x>
void func() { }

int main() {
  func<__integer_pack(10)...>();  
}
```
```
$ g++ integer_pack.cxx
integer_pack.cxx: In function ‘int main()’:
integer_pack.cxx:5:25: error: use of built-in parameter pack ‘__integer_pack’ outside of a template
    5 |   func<__integer_pack(10)...>();
      |                         ^

$ circle integer_pack.cxx
<okay>
```

Circle takes the pack-yielding core of `__integer_pack` and positions it as a first-class language feature, by exposing a "pack index" mechanism noted with `int...`:

* `int...` - yields the index of the current element in the pack expansion.
* `int...(count)` - generates an int-valued parameter pack of _count_ elements.
* `int...(begin:end:step)` - generates an int-valued parameter pack from extended slice notation. All terms are optional, however an `end` term is required to compute the pack size if it cannot be inferred from other bounded parameter packs in the expansion. 

[**pack.cxx**](pack.cxx)
```cpp
#include <tuple>
#include <iostream>

int main() {
  auto tuple = std::make_tuple('a', 2, 3.3);
  std::cout<< int...<< ": "<< tuple...[:]<< "\n" ...;
}
```
```
$ circle pack.cxx && ./pack
0: a
1: 2
2: 3.3
```

`int...`, when written without an attached range or slice, yields back the index of the current pack expansion element. This is useful when the size of the parameter pack is dictated by another expression inside the same expansion. In this case, the expression that sets the pack size is the `tuple..[:]` slice.

[**pack2.cxx**](pack2.cxx)
```cpp
#include <iostream>

void print_values(const auto&&... x) {
  std::cout<< "{";
  if constexpr(sizeof...(x))
    std::cout<< " "<< x...[0];
  std::cout<< ", "<< x...[1:] ...;
  std::cout<< " }\n";
}

template<int N>
void func() {
  print_values(int...(N)...);
}

int main() {
  // Legacy gcc __integer_pack intrinsic. This is how std::integer_sequence 
  // is actually implemented;
  print_values(__integer_pack(5)...);

  // New int...() expression. It can be given a count...
  print_values(int...(5)...);

  // ... Or it can be given a slice.
  // Print the odds between 1 and 10.
  print_values(int...(1:10:2)...);

  // Print a countdown from 9 to 0. When the step is negative, the 
  // begin index is exclusive and the end index is inclusive.
  print_values(int...(10:0:-1)...);

  struct obj_t {
    int x, y, z;
  };
  obj_t obj { 100, 200, 300 };
  std::cout<< int...(1:)<< ": "<< obj...[:]<< "\n" ...;
}
```
```
$ circle pack2.cxx && ./pack2
{ 0, 1, 2, 3, 4 }
{ 0, 1, 2, 3, 4 }
{ 1, 3, 5, 7, 9 }
{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
1: 100
2: 200
3: 300
```

The `int...(count)` expression generates an `int` parameter pack with values from 0 to count-1. `int...(begin:end:step)` is a pack-yielding slice. `end` is an optional parameter and the size of the pack may be inferred from other packs in the same expansion. In this mode, it acts like a linear equation around `int...`, where the `begin` and `step` terms are constant and linear adjustments.

## Transforming objects.

[**transform.cxx**](transform.cxx)
```cpp
#include <tuple>
#include <iostream>

typedef float __attribute__((vector_size(16))) vec4;

template<typename a_t, typename b_t, typename func_t>
void binary_op(a_t& a, const b_t& b, func_t f) {
  f(a...[:], b...[:]) ...;
}

int main() {
  // Add a tuple with a vector.
  std::tuple<int, float, double, long> left(1, 2.f, 3.0, 4ll);
  vec4 right(1, 2, 3, 4);

  // Can pass a lambda and let an algorithm destructure the arguments.
  binary_op(left, right, [](auto& a, auto b) {
    a += b;
  });

  // Or just do it directly in line.
  left...[:] += right...[:] ...;

  std::cout<< int...<< ": "<< left...[:]<< "\n"...;
}
```

Universal member access permits defining operations that work over heterogeneous inputs. Adding the elements of a vector into the elements of a tuple, compiles out of the box, and generic lambdas accommodate the changing element types of the tuple.

[**transform2.cxx**](transform2.cxx)
```cpp
#include <iostream>
#include <tuple>
#include <utility>
#include <algorithm>

// Re-arrange the tuple by element size.
// Elements with the smallest size are sorted to go first in result object.
auto sort_tuple(const auto& tuple) {
  // Sort once per template instantiation. .first is the sizeof the element.
  // .second is the gather index.
  @meta std::pair<int, int> sizes[] { 
    std::make_pair(sizeof(tuple.[:]), int...) ...
  };
  @meta std::sort(sizes + 0, sizes + sizeof. sizes); 

  // The gather operation. ...[] gathers from tuple. sizes...[:].second is the
  // gather index for each output.
  return std::make_tuple(tuple.[sizes.[:].second] ...);
}

int main() {  
  auto tuple = std::make_tuple(1, 2.f, '3', 4ll, 5.0, 6);
  auto tuple2 = sort_tuple(tuple);

  std::cout<< decltype(tuple2).string << "\n";
  std::cout<< tuple2.[:]<< " (size = "<< sizeof(tuple2.[:])<< ")\n" ...;
}
```
```
$ circle transform2.cxx && ./transform2
std::tuple<char, int, float, int, long long, double>
3 (size = 1)
1 (size = 4)
2 (size = 4)
6 (size = 4)
4 (size = 8)
5 (size = 8)
```

Universal member access pairs well with Circle's `@meta`-driven imperative metaprogramming. This sample restructures an std::tuple into a new std::tuple where the members are ordered by increasing size.

`sort_tuple` fills a compile-time array `sizes` with pairs holding the size of each tuple element, and its gather index, indicated by `int...`, which is the current pack expansion index. `std::sort` re-orders the pairs by increasing size, and when the sizes are the same, by increasing gather index, which guarantees stability.

```cpp
  return std::make_tuple(tuple...[sizes...[:].second] ...);
```

This statement is brings together many of the mechanisms described in this document. `tuple.[]` subscripts the original tuple. But we want a full gather operation, not a single subscript, so we feed it with `sizes.[:].second`, which slices the `sizes` array into a parameter pack of gather indices. Expanding `tuple.[sizes.[:].second]` into `std::make_tuple`s arguments list coins a new tuple, with members sorted according to size.
