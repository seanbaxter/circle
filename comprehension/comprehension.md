# List comprehensions, slices, ranges, for-expressions, functional folds and expansion expressions.

1. [Dynamic pack generators](#1-dynamic-pack-generators)  
  a. [Slice expressions](#a-slice-expressions)  
  b. [The `@range` operator](#b-the-range-operator)  
  c. [For-expressions](#c-for-expressions)  
1. [Dynamic pack consumers](#2-dynamic-pack-consumers)  
  a. [Expansion expressions](#a-expansion-expressions)  
  b. [List comprehensions](#b-list-comprehensions)  
  c. [Functional fold expressions](#c-functional-fold-expressions)  
  d. [For-range-initializers](#d-for-range-initializers)  
1. [Modifiers](#3-modifiers)  
  a. [Sequences](#a-sequences)  
1. [Static slice expressions](#static-slice-expressions)  
  a. [Static slices on template parameter packs](#static-slices-on-template-parameter-packs)  
  b. [Static slices on tuple-like objects](#static-slices-on-tuple-like-objects)  
1. [Circle vs C++ ranges](#circle-vs-c-ranges)  
  a. [Hello, Ranges!](#hello-ranges)  
  b. [any_of, all_of, none_of](#any_of-all_of-none_of)  
  c. [count](#count)  
  d. [count_if](#count_if)  
  e. [for_each on sequence containers](#for_each-on-sequence-containers)  
  f. [for_each on associative containers](#for_each-on-associative-containers)  
  g. [is_sorted](#is_sorted)  
  h. [Filter and transform](#filter-and-transform)  
  i. [Generate ints and accumulate](#generate-ints-and-accumulate)  
  j. [Convert a range comprehension to a vector](#convert-a-range-comprehension-to-a-vector)  
1. [Points of evolution](#points-of-evolution)  

Circle adds a new property to all expressions: the _dynamic pack_ property. Dynamic packs resemble _static packs_ (such as those bound to variadic template parameters) in that they are lazily threaded through enclosing expressions and are expanded with the `...` token. Dynamic packs, however, represent entities with dynamic sizes, and expansion causes the generation of implicit loops.

The dynamic pack property enables powerful features that productivity languages like Python and Matlab, as well as many functional languages, have offered for decades. C++ has never implemented these features, and they haven't been part of the C++ user's consciousness, because C++ has stayed away from adding features with a dynamic runtime component. That is, most C++ features have enhanced only the compiler front-end, and kept code-generation capability that doesn't deviate much from that of C. Top-line features like class inheritance, templates, parameter packs, lambda functions, type inference, modern value categories and concepts/requires-clauses don't generate code at runtime; instead they offer to reshape C++ source code to more concisely generate the same kind of executable.

C++ has incorporated a couple of dynamic features, but only virtual functions (the cheapest of these features) and exception handling are widely used. RTTI and virtual inheritance have high costs, low utility, and feel unidiomatic and weird. None of these dynamic features are at the heart of the C++ user experience.

The dynamic features introduced here are all about productivity. Their use generates real code, like heap allocations and loops. They all target collections, including builtin arrays (which are necessarily statically sized but are attacked with dynamic loops) and STL containers. But instead of traversing the collection with an explicit loop or with a call to an STL algorithm like `std::transform`, you visit the elements with a single use of the `...` pack expansion token.

```cpp
std::vector<int> v { 5, 3, 1, 4, 2, 3, 5, 1 };
printf("%d\n", v[:])...;
```

This code prints all elements in a vector container, each element on its own line. `...` is a pack expansion, so it needs to operate on a pack expression. In this case, `printf("%d\n", v[:])` is a _pack expression_, and `v[:]` is a _pack generator_. `[:]` is a _slice operator_, which transforms the contents of an array or STL container into a dynamic parameter pack.

Similarly to Python, the slice operator takes three optional arguments:
* **begin** - a signed value indicating where to start sampling the container. When negative, the index counts from the end of the container. -1 points to one past the last element.
* **end** - a signed value indicating where to stop sampling the container.
* **step** - a non-zero signed value indicating how large a step to take between samples. A negative step orders right-to-left visitation.

```cpp
std::vector<int> v { 5, 3, 1, 4, 2, 3, 5, 1 };
printf("%d\n", v[::-1])...;
```

This code prints all elements in reverse order. If the step is positive (it defaults to 1 when not specified), the begin and end indices default to 0 and -1, respectively. If the step is negative, the begin and end indices default to -1 and 0. The above slice is shorthand for `v[-1:0:-1]`. 

```cpp
std::vector<int> v { 5, 3, 1, 4, 2, 3, 5, 1 };
printf("%d\n", v[v.size()/4:-v.size()/4])...;
```

This code prints the elements in the middle of the vector. We start 1/4 from the beginning, and stop when we're 1/4 from the end.

```cpp
std::vector<int> a { 5, 3, 1, 4, 2, 3, 5, 1 }, b { 2, 1, 4, 6, 2, 0, 9, 5 };
a[:] = 3 * a[:] + b[:] ...;
```

This statement replaces each element in `a` with the element-wise computation `3 * a + b`. Note that `std::vector` itself doesn't overload operators * or +. Operations on pack expressions transform not the containers but the elements themselves. The result object of the slice expression `a[:]` is `lvalue int`, so we can apply any builtin operator on the slice, or call any function on it. The magic that enables dynamic loop generation is in the dynamic pack bit.

```cpp
std::vector<int> v { 5, 3, 1, 4, 2, 3, 5, 1 };
std::vector<int> v2 = [ v[::-1]... ];
```

As in Python, square brackets form a list comprehension expression. The result object has type `std::vector<T>`, where the argument T is inferred from the type of the elements in the list. List comprehensions provide expansion loci for dynamic packs, so we can expand the contents of `v` into `v2`. At runtime, the executable initializes a vector, reserves memory, and expands the dynamic pack into the container.

```cpp
std::vector<int> a { 5, 3, 1, 4, 2, 3, 5, 1 }, b { 2, 1, 4, 6, 2, 0, 9, 5 };
std::vector<int> c = [ 3 * a[:] + b[:]... ];
```

This code performs and element-wise `3 * a + b` computation and stores each result into a new vector, `c`. 

```cpp
std::vector<int> a { 5, 3, 1, 4, 2, 3, 5, 1 }, b { 2, 1, 4, 6, 2, 0, 9, 5 };
std::set<int> c = [ 3 * a[:] + b[:]... ];
```

Here we initialize an `std::set` with the results of an expansion. The result object of the list comprehension is `std::vector<int>`. As `std::set<int>` doesn't have a constructor that takes a vector, we'd normally expect a compiler error. Circle list comprehensions, however, implicitly convert to `std::initializer_list` when they fail to initialize the left-hand side. The data in the vector serves as backing store for the `std::initializer_list`, which is simply a pointer into the vector and a count (that is, `vec.data()` and `vec.size()`). This implicit conversion lets us initialize or assign to any STL or user-defined container equiped with an `std::initializer_list` constructor or assignment operator. The vector that holds the list comprehension is freed at the end of the initialization, since that storage is no longer required (the container having copied the initializer list into its internal data format).

Uniform initialization in Standard C++ is incomplete. The `std::initializer_list` type itself is dynamic--the length of the structure is stored in an opaque data member; it's not part of the type. But Standard C++ does nothing to exploit this dynamicness of size! The only way to generate an `std::initializer_list` is with braced initializers, and those only support a static count of elements. Why? Well, Standard C++ doesn't like features that make heap allocations, and that's what's required to allocate backing store when the number of elements in an initializer list is not known at compile time.

By creating and reserving heap memory for an `std::vector`, Circle list comprehension builds this backing store for `std::initializer_list`. Additionally, the resulting initializer list *does not* participate in uniform initialization, so there's no risk of binding to a non-`initializer_list` constructor when initializing from list comprehension. The declaration either uses an `std::vector` constructor or assignment, or failing that, an `std::initializer_list` constructor or assignment.

```cpp
std::vector<int> a { 5, 3, 1, 4, 2, 3, 5, 1 }, b { 2, 1, 4, 6, 2, 0, 9, 5 };
int sum = (... + a[:]);
int greater = (... + (int)(a[:] > b[:]));
bool has_equal = (... || (a[:]==b[:]));
int max = (... + std::max a[:]);
```

Circle also extends _fold-expressions_ to participate in dynamic pack expansion. We can provide a binary operator or two-parameter function and iteratively apply it to each element in the dynamic pack expression, reducing the results into the initializer object. `sum` simply adds up all the elements in a. `greater` compares each pair of elements, and increments the counter when the comparison is true. `has_equal` is set to true if any corresponding elements have the same value. `max` holds the largest value in `a`.

We don't have to write any loop. We don't have to call any STL algorithm. We can express our reduction using an existing but under-utilized syntax.

[**iterate.cxx**](iterate.cxx)
```cpp
#include <algorithm>
#include <cstdio>

int main() {
  int x = 10;
  std::vector<int> v { 5, 3, 1, 4, 2, 3, 5, 1 };
  std::vector<int> u { 7, 1, 2, 2, 4, 3, 8, 7 };

  // 1) Print x * v for each element in v. Use std::for_each.
  std::for_each(v.begin(), v.end(), [=](int y) { printf("%2d ", x * y); });
  printf("\n");

  // Print x * v for each element in v using Circle dynamic packs.
  printf("%2d ", x * v[:])...; printf("\n");

  // 2) Print x * v in reverse order.
  std::for_each(v.rbegin(), v.rend(), [=](int y) { printf("%2d ", x * y); });
  printf("\n");

  // Use a step of -1 to visit the elements in reverse order.
  printf("%2d ", x * v[::-1])...; printf("\n");

  // 3) Use any_of to confirm a number greater than 4.
  bool is_greater = std::any_of(v.begin(), v.end(), [](int x) { return x > 4; });
  printf("STL: greater than 4? %s\n", is_greater ? "true" : "false");

  // Use Circle dynamic packs to confirm a number greater than 4.
  bool is_greater2 = (... || (v[:] > 4));
  printf("Circle: greater than 4? %s\n", is_greater2 ? "true" : "false");

  // 4) Print u * v. How do we do this with STL algorithms? Do we need
  // boost::zip_iterator to present two simultaneous views as one?

  // With Circle, just use two slices.
  printf("%2d ", u[:] * v[:])...; printf("\n");

  // 5) Print the sum of odds and evens. That is, print v[0] + v[1], 
  // v[2] + v[3], etc.
  // Do we need to combine a step_iterator with a zip_iterator? What is the
  // C++ answer?

  // With Circle, use the step argument of two slices.
  printf("%2d ", v[::2] + v[1::2])...; printf("\n");
}
```
```
$ circle iterate.cxx  && ./iterate 
50 30 10 40 20 30 50 10 
50 30 10 40 20 30 50 10 
10 50 30 20 40 10 30 50 
10 50 30 20 40 10 30 50 
STL: greater than 4? true
Circle: greater than 4? true
35  3  2  8  8  9 40  7 
 8  5  5  6 
```

Circle's dynamic packs are about vectorizing operations over collections. C++ recognizes the power of these vectorized expressions, so it first introduced STL algorithms that take predicate objects to carry logic from the user to the library. C++11 added lambda expressions to ease the creation of funciton objects. C++20 added ranges to make the algorithms more composable. These features all paper over a fundamental problem: C++ doesn't let the user directly express their intent. 

* Why should the user have to choose an option from a menu of algorithms then wrap their logic in a lambda? Lambda closure involves making a lot of subtle decisions, which are all avoided when using the dynamic pack expansion approach.
* Why introduce algorithms that have a fixed interface then require the user to conform to this interface by means of programmable iterators?
* Why must we rely on libraries to emulate functionality that has been built into Fortran, Matlab, Python and other languages for decades?

C++11's parameter packs showed us the way forward: add a pack flag to each expression in the language, allowing lazy evaluation and a loop over all elements once the enclosing expansion ellipsis is struck. Circle extends the parameter pack philosophy past simple template parameter packs and onto containers with dynamic data.

Circle's first cut into dynamic packs adds features that break down in three composable categories:

## 1. Dynamic pack generators

Special operators yield dynamic pack expressions. The slice operator, `[:]`, yields a pack over the contents of an array or STL container. Expressions involving dynamic pack expressions are themselves dynamic pack expressions. For example, `sq(x[:] + 1)` is a pack expression--the result of adding a pack with a non-pack is a pack, and passing a pack as a function argument makes the result object of the function call a pack. This is the same logic as static parameter packs in C++11.

Dynamic packs, as with their parameter pack predecessors, must be expanded. Dynamic pack expansion generates an implicit runtime loop over each element in the sequence.

[**pack1.cxx**](pack1.cxx)
```cpp
#include <vector>
#include <cstdio>

inline int sq(int x) {
  return x * x;
}

int main() {
  std::vector x = [ @range(10)... ];      // shorthand for @range(0:10:1).
  std::vector y = [ @range(10::-1)... ];  // shorthand for @range(10:0:-1).

  // Compute element-wise sq(x) + 5 * y, 
  std::vector z = [ sq(x[:]) + 5 * y[:] ... ];

  // Print each element.
  printf("sq(%d) + 5 * %d -> %2d\n", x[:], y[:], z[:]) ...;
}
```
```
$ circle pack1.cxx 
$ ./pack1
sq(0) + 5 * 9 -> 45
sq(1) + 5 * 8 -> 41
sq(2) + 5 * 7 -> 39
sq(3) + 5 * 6 -> 39
sq(4) + 5 * 5 -> 41
sq(5) + 5 * 4 -> 45
sq(6) + 5 * 3 -> 51
sq(7) + 5 * 2 -> 59
sq(8) + 5 * 1 -> 69
sq(9) + 5 * 0 -> 81
```

This test demonstrates three instances of list comprehension and one expansion expression statement. Dynamic pack expansion is a major labor-saving feature. Consider what the compiler actually generates to implement the above:

[**pack2.cxx**](pack2.cxx)
```cpp
#include <vector>
#include <algorithm>
#include <cstdio>

inline int sq(int x) {
  return x * x;
}

int main() {
  // std::vector = [ @range(10)... ];
  std::vector<int> x;
  {
    // The expansion count is inferred from the range expression.
    size_t count = 10;

    // Declare an object for the current index in the range.
    int begin = 0;

    // Loop until we've exhausted the range.
    while(count--) {
      x.push_back(begin);

      // The begin index for positive step is inclusive. Decrement it at
      // the end of the loop.
      ++begin;
    }
  }

  // std::vector y = [ @range(10::-1)... ];
  std::vector<int> y;
  {
    size_t count = 10;
    int begin = 10;
    while(count--) {
      // The begin index for negative step is exclusive. Decrement it at
      // the start of the loop.
      --begin;

      y.push_back(begin);
    }
  }

  // std::vector z = [ sq(x[:]) + 5 * y[:] ... ];
  std::vector<int> z;
  {
    // Find the size for each slice in the pack expression.
    size_t x_count = x.size();
    size_t y_count = y.size();

    // Use the minimum slice size to set the loop count.
    size_t count = std::min(x_count, y_count);

    // Declare iterators for the current item in each slice.
    auto x_begin = x.begin();
    auto y_begin = y.begin();

    while(count--) {
      z.push_back(sq(*x_begin) + 5 * *y_begin);

      // Both slices have an implicit +1 step, so perform post-step increment.
      ++x_begin;
      ++y_begin;
    }
  }

  // printf("sq(%d) + 5 * %d -> %2d\n", x[:], y[:], z[:]) ...;
  {
    // Find the size for each slice in the pack expression.
    size_t x_count = x.size();
    size_t y_count = y.size();
    size_t z_count = z.size();
    size_t count = std::min(std::min(x_count, y_count), z_count);

    auto x_begin = x.begin();
    auto y_begin = y.begin();
    auto z_begin = z.begin();

    while(count--) {
      printf("sq(%d) + 5 * %d -> %2d\n", *x_begin, *y_begin, *z_begin);

      ++x_begin;
      ++y_begin;
      ++z_begin;
    }
  }
}
```

Why query `size` on each slice rather than using a `begin != end` predicate like ranged _for-statetment_? We want to generate fast code, and computing an expansion count lets us get away with evaluating just a single predicate expression to enter each loop step, no matter how complex the expression. This code is performant but verbose. Circle generate this code for you from a concise dynamic pack syntax.

### a. Slice expressions

Equivalent to the slice syntax of Python. `v[begin:end:step]` yields a dynamic pack that visits the elements of container `v`, starting at offset `begin`, ranging through to offset `end`, and incrementing by `step` elements at each step. The three operands and the second colon are optional. `v[:]` iterates all elements of `v` in forward order. `v[::-1]` iterates the elements in reverse order.

[**slice.cxx**](slice.cxx)
```cpp
#include <string>
#include <algorithm>
#include <cstdio>

int main() {
  std::string s = "Hello world";
  printf("%c ", s[:])...;
  printf("\n");         // Prints 'H e l l o   w o r l d '

  // Loop over the first half of s in forward order and the back half of 
  // s in reverse order, swapping each pair.
  size_t mid = s.size() / 2;
  std::swap(s[:mid:1], s[:mid:-1])...;
  puts(s.c_str());      // Prints 'dlrow olleH'

  // Reverse it a second time. 
  // The :1 forward step is implicit so is dropped. The end iterator on 
  // the back half is dropped, because the expansion expression's loop count
  // is inferred from the shortest slice length. 
  std::swap(s[:mid], s[::-1])...;
  puts(s.c_str());      // Prints 'Hello world'

  // Reverse the string using list comprehension.
  std::string s2 = [s[::-1]...];
  puts(s2.c_str());     // Prints 'dlrow olleH'

  // Print the front half in forward order and the back half in reverse order.
  std::string s3 = [s[:mid]..., s[:mid:-1]...];
  puts(s3.c_str());     // Prints 'Hellodlrow '

  // Use list comprehension to collect the even index characters, then
  // the odd index characters. Uppercase the front half and lowercase the
  // back half.
  std::string s4 = [(char)toupper(s[::2])..., (char)tolower(s[1::2])...];
  puts(s4.c_str());     // Prints 'HLOWRDel ol'
}
```
```
$ circle slice.cxx
$ ./slice
H e l l o   w o r l d 
dlrow olleH
Hello world
dlrow olleH
Hellodlrow 
HLOWRDel ol
```

Slices are implicitly sized by querying the `size` member function on the container. After adjusting by the `begin` and `end` indices, the size is divided by `step`, yielding a slice element count. The length of the dynamic loop generated at the expansion locus is the minimum of each of the slice counts. This convenience relieves us from having to over-specify ranges, as they can be cooperatively inferred.

When the step counter is positive, the begin index is _inclusive_ and the end index is _exclusive_. That is, `v[5:10:1]` visits elements 5 through 9, but not 10. 

Negative step counters change the slice semantics in an important way. Here, the begin index is _exclusive_ and the end index is _inclusive_. This differs from the Python convention for extended slice operators, which is defective in that it provides no way to address the full range of a container using a negative step index. `v[10:0:-1]` visits elements 9 through 0 in descending order.

Negative begin and end indices indicate steps from the end of the container. -1 means one past the last element in the container, which corresponds with the `.end()` accessor in STL containers. `v[0:-1:1]` visits all members in the container in forward order: -1 codes to one past the last element, and since (for positive steps) the end index is exclusive, we visit the last element, but no further.

`v[-1:0:-1]` visits all members in the container in reverse order. -1 codes to one past the the last element. When the step size is negative, the begin index is exclusive, so expansion actually starts at the last element and continues on towards decreasing indices. Python has the wrong inclusive/exclusive treatment of indices for negative step sizes, creating an addressing singularity.

It's critical to understand that a slice by itself does not create a temporary object. The result object of a slice expression is the result object of `*v.begin()`, which is usually an lvalue expression of the container's type. You can incorporate this result object into a larger expression like you would any other entity in C++: type and value category conversions operate as expected; you can take the address of the lvalue, you can pass it to other functions, and so on. Only when you hit the expansion locus `...` does the expression get realized into code. Expansion in a fold expression reduces the range expression into a single value. Expansion in a list comprehension constructs an `std::vector`. Expansion in an expression statement converts to void (discarding the result object) and generates an implicit loop over each slice element.

### b. The `@range` operator

Like a slice over the integers. `@range()` is the infinite sequence of ascending integers starting at 0. `@range(10)` is the set of integers between 0 and 9. `@range(5:10)` is the set of integers between 5 and 9. `@range(::2)` is all the even non-negative integers.

The `@range` operator does not perform negative index mapping like slices do. That is, `@range(-1:10)` yields a pack expression that loops from -1 to 9 when expanded. Ranges may also be unsized, which is okay if combined with other range or slice expressions which do have a size, or when modified with a _take-clause_. As long as a pack count can be inferred, unsized range expressions are permitted.

[**range.cxx**](range.cxx)
```cpp
#include <vector>
#include <string>
#include <cstdio>

int sq(int x) { return x * x; }

int main() {
  printf("%d ", @range(10))...; 
  printf("\n");        // Prints '0 1 2 3 4 5 6 7 8 9 '

  printf("%d ", @range(5:25:5))...;
  printf("\n");        // Prints '5 10 15 20 '

  printf("%d ", @range(25:5:-5))...;
  printf("\n");        // Prints '24 19 14 9'

  // Sum up integers from 0 to 9.
  int sum = (... + @range(10));

  // Sum up squares of integers from 0 to 9.
  int sum_squares = (... + sq(@range(10)));

  // Fill two vectors with ints.
  std::vector v1 = [@range(3:18:3)...];   // 3, 6, 9, 12, 15
  std::vector v2 = [@range(5:15:2)...];   // 5, 7, 9, 11, 13

  printf("%d ", v1[:])...; printf("\n");
  printf("%d ", v2[:])...; printf("\n");

  // Get their dot product.
  double dot = (... + (v1[:] * v2[:]));
  printf("%f\n", dot);

  // Get their L2 norm.
  double l2 = sqrt(... + sq(v1[:] - v2[:]));
  printf("%f\n", l2);

  // Fill array with strings.
  const char* days[] {
    "Sunday", "Monday", "Tuesday", "Wednesday", 
    "Thursday", "Friday", "Saturday"
  };
  // Print index/string pairs.
  printf("%d: %s\n", @range(1:), days[:])...;

  // Prints:
  // 1: Sunday
  // 2: Monday
  // 3: Tuesday
  // 4: Wednesday
  // 5: Thursday
  // 6: Friday
  // 7: Saturday
}
```

The range operator uses the same `begin:end:step` syntax as slices, but as it pulls from the set of integers rather than elements in a container, there is no size adjustment for negative indices. `@range` is a cheap way to generate a finite or infinite sequence of integers. It corresponds exactly to the [range](https://docs.python.org/3/library/functions.html#func-range) function in Python 3.

`@range` is useful when paired with a slice expression, as it provides the current index for the slice element (with any begin offset and step size desired). 

### c. For-expressions

A streamlined take on the ranged _for-statement_, which may be used from list comprehensions or fold expressions. This allows us to bind a declaration to each step in a loop, add an optional filter (with an _if-clause_), and emit elements to the dynamic pack consumer.

```
for-expression:
  for [index-name, ] [ref-qual] [decl-name | [structured-binding] ] : for-initializer [if condition] => body
```

The _for-expression_'s syntax is rather more Pythonic than the syntax for ranged a _for-statement_. The parentheses are dropped, because they aren't needed. The type-specifier is dropped, and the placeholder type `auto` is assumed. An optional _ref-qualifier_ `&` or `&&` binds a reference to the initializer rather than an object type. A colon separates the loop declaration from its initializer. Following on the initializer's heels is an optional _if-filter_, introduced with the `if` keyword. The fat arrow `=>` should be read "then." That introduces the body of the _for-expression_.

The _for-expression_ supports an optional index name. This corresponds exactly to the [enumerate](https://docs.python.org/3/library/functions.html#enumerate) function in Python 3. It yields the slice index along with the slice result object at each step.

If the _for-initializer_ of a _for-expression_ is an integer, the expression loops from 0 to that integer value. This is visually cleaner than expanding a `@range` expression to generate indices in the _for-initializer_. This feature has also been extended to C++ ranged _for-statements_.

[**for.cxx**](for.cxx)
```cpp
#include <string>
#include <vector>
#include <cstdio>

int main() {
  std::string s = "Hello world";

  // Use a for-expression to print only the lowercase characters.
  std::string s2 = ['*', for c : s if islower(c) => c..., '*'];
  puts(s2.c_str());     // Prints '*elloworld*'

  // Use a for-expression to double each character.
  std::string s3 = [for c : s => { c, c }...];
  puts(s3.c_str());     // Prints 'HHeelllloo  wwoorrlldd'

  // Use a for-expression to emit upper/lower-case pairs.
  std::string s4 = [for c : s => {(char)toupper(c), (char)tolower(c)}...];
  puts(s4.c_str());     // Prints 'HhEeLlLlOo  WwOoRrLlDd'

  // Use the index to create alternating upper and lowercase characters.
  std::string s5 = [for i, c : s => (char)((1&i) ? tolower(c) : toupper(c))...];
  puts(s5.c_str());     // Prints 'HeLlO WoRlD'

  // Create a vector of vectors.
  printf("Creating a vector of vectors:\n");
  std::vector vecs = [for i : 5 => [for i2 : i => i...] ...];
  for(auto& v : vecs) {
    printf("[ "); printf("%d ", v[:])...; printf("]\n");
  }
}
```
```
$ circle ranges/for.cxx
$ ./for
*elloworld*
HHeelllloo  wwoorrlldd
HhEeLlLlOo  WwOoRrLlDd
HeLlO WoRlD
Creating a vector of vectors:
[ ]
[ 1 ]
[ 2 2 ]
[ 3 3 3 ]
[ 4 4 4 4 ]
```
A list comprehension is composed of one or more _initializer-clauses_, and any of those clauses may be an expanded range/slice expression or expanded _for-expression_. The `s2` example expands a _for-expression_ between two asterisks. The attached filter only emits the body expression if the character is lower case. In the `s3` example, a sequence modifier emits a pair of characters into the list comprehension. `s4` is constructed by emitting a sequence with both uppercase and lowercase versions of the iterated character into the list. Finally, `s5` uses an index to alternate lower and uppercase characters on even and odd steps.

As with slices, keep in mind that _for-expressions_ are lazy. The result object of the body is only evaluated from a loop generated by the compiler at the expansion locus. _for-expressions_ are more constrained than other kinds of expressions. Their expansion locus is always immediately following the loop's body, on the same level as the `for` keyword. This prevents confusing uses of _for-expressions_, like occurrences as function arguments, or expansions outside the function call. That capability is still available, but the _for-expression_ machinery to the left of the body must be coordinated with the ellipsis token on the right.

## 2. Dynamic pack consumers

Dynamic pack expressions must be expanded, and the loci of these expansions occur in _dynamic pack consumers_. The evolution plans of Circle involve adding composible dynamic pack generates and dynamic pack consumers, creating a multiplicative increase in language capability.

Whenever a dynamic pack is used, the pack must be expanded with the ellipsis token. These points of expansion are called _loci_, and currently are supported by three consumers: list comprehensions, fold expressions and expansion expressions.

[**locus.cxx**](locus.cxx)
```cpp
#include <vector>
#include <cstdio>

int main() {
  
  // Create a vector of vectors, but use different expansion loci.
  auto m1 = [ [ @range(5)... ] ];
  auto m2 = [ [ @range(5) ]... ];

  printf("m1:\n[\n");
  for(auto& v : m1) {
    printf("  [ "); printf("%d ", v[:])...; printf("]\n");
  }
  printf("]\n\n");

  printf("m2:\n[\n");
  for(auto& v : m2) {
    printf("  [ "); printf("%d ", v[:])...; printf("]\n");
  }
  printf("]\n");
}
```
```
$ circle locus.cxx 
$ ./locus
m1:
[
  [ 0 1 2 3 4 ]
]

m2:
[
  [ 0 ]
  [ 1 ]
  [ 2 ]
  [ 3 ]
  [ 4 ]
]
```

No matter which dynamic pack consumer is chosen, the choice of expansion loci effects the structure of the result. This example shifts the expansion token to yield two different `vector<vector<int>>`s. In the first example, the expansion is in the inner list comprehension, so the inner dimension is 5, and the outer dimension is 1. In the second example, the expansion is in the outer list comprehension, turning the inner comprehension into an unexpanded dynamic pack expression (that is, it has a prvalue `vector<int>` result object with the dynamic pack bit set), which gets expanded into the outer comprehension. The result is five inner vectors, each with a single scalar element.

### a. Expansion expressions

Write a slice expression and expand it using the ellipsis at the end of the expression statement. The result object is converted to void and discarded (like any other expression statement), and an implicit loop is emitted which replays the expression, simultaneously stepping through the elements of slices and indices of ranges.

[**expansion.cxx**](expansion.cxx)
```cpp
#include <vector>
#include <set>
#include <algorithm>
#include <cstdio>

int main() {
  std::vector<int> v { 4, 2, 2, 2, 5, 1, 1, 9, 8, 7 };

  // Print the vector values.
  printf("%3d ", v[:])...; printf("\n");

  // Fill the vector with powers of 2.
  v[:] = 1<< @range()...;
  printf("%3d ", v[:])...; printf("\n");

  // Populate a set with the same values. Print their values.
  // Slice is like an enhanced ranged-for, so it supports the usual
  // STL containers, or anything else with begin, end and size member
  // functions.
  std::set<int> set;
  set.insert(v[:])...;
  printf("%3d ", set[:])...; printf("\n");

  // Add up all values into an accumulator. This is better done with a
  // fold expression.
  int sum = 0;
  sum += v[:]...;
  printf("sum = %d\n", sum); // sum = 41

  // Add each right element into its left element. Because the loop is
  // executed left-to-right, we don't risk overwriting any elements before
  // we source them.
  v[:] += v[1:]...;
  printf("%3d ", v[:])...; printf("\n"); //  6  4  4  7  6  2 10 17 15  7 

  // Reset the array to 1s.
  v[:] = 1...;

  // Perform a prefix scan. Add into each element the sum of all elements
  // before it. This is like the fourth example, but with the operands 
  // flipped.
  v[1:] += v[:]...;
  printf("%3d ", v[:])...; printf("\n"); //  1  2  3  4  5  6  7  8  9 10

  // Reverse the array in place. Exchange each element with its mirror
  // up through the midpoint.
  int mid = v.size() / 2;
  std::swap(v[:mid], v[::-1])...;
  printf("%3d ", v[:])...; printf("\n"); // 10  9  8  7  6  5  4  3  2  1

  // Add into each element its index from the range of integers.
  v[:] += @range()...;
  printf("%3d ", v[:])...; printf("\n"); // 10 10 10 10 10 10 10 10 10 10

  // Reset the array to ascending integers. Now swap the even and odd
  // positions. The 2-element step skips every other item.
  v[:] = @range()...;
  std::swap(v[::2], v[1::2])...;
  printf("%3d ", v[:])...; printf("\n"); //  1  0  3  2  5  4  7  6  9  8
}
```

Expansion expressions are real time savers. Not only do they make the loop implicit, but the contained slices perform the intricate index calculations that are a common source of errors in imperative programming. The implicit loop visits only those elements that are defined across each slice in the expression. The statement `v[1:] += v[:]...` to perform a prefix scan, for example, starts at index 1 on the left-hand side and 0 on the right-hand side. Therefore, the left-hand side slice has one fewer element, so the implicit loop count is adjusted to `v.size()-1` to prevent out-of-bounds access violations.

The slices are a big improvement on ranged _for-statements_, providing begin and end indices as well as a step count and direction, for any number of containers simultaneously.

### b. List comprehensions

The most exciting new feature is list comprehension, which collects an _initializer-list_ into an `std::vector` at runtime. List comprehension may be used to initialize any STL type with an `std::initializer_list` constructor, by materializing the `std::vector` result object and using it as a backing store for the `std::initializer_list`s temporaries.

As with [list comprehensions in Python](https://docs.python.org/3/whatsnew/2.0.html#list-comprehensions), they are introduced in Circle as a list of initializers inside square brackets `[ ]`. This syntax is not ambiguous with lambda function expressions: there, the square brackets must be followed by `(`, `<`, `->` or `{`. Square brackets followed by any other token matches the list comprehension grammar.

[**list_comp.cxx**](list_comp.cxx)
```cpp
#include <vector>
#include <cstdio>

int main() {
  std::vector<int> v { 3, 1, 2, 5, 3, 4, 4, 7, 6 };

  // For each odd number in v, repeat that number that many times.
  // Prints '3 3 3 1 5 5 5 5 5 3 3 3 7 7 7 7 7 7 7'
  std::vector<int> v2 = [for i : v if 1 & i => for x : i => i... ... ];
  printf("%d ", v2[:])...; printf("\n");

  // Cut off the same comprehension after 10 elements.
  // Prints '3 3 3 1 5 5 5 5 5 3'
  std::vector<int> v3 = [for i : v if 1 & i => for x : i => i... ... ] | 10;
  printf("%d ", v3[:])...; printf("\n");

  // Interleave each element of v3 with 0.
  std::vector<int> v4 = [ { v3[:], 0 }... ];
  printf("%d ", v4[:])...; printf("\n");
  
  // Create a triangular structure of vectors.
  auto v5 = [ for i : 5 => [ for x : i => i ... ] ... ];
  for(auto& v : v5) {
    printf("[ "); printf("%d ", v[:])...; printf("]\n");
  }
}
```
```
$ circle list_comp.cxx 
$ ./list_comp
3 3 3 1 5 5 5 5 5 3 3 3 7 7 7 7 7 7 7 
3 3 3 1 5 5 5 5 5 3 
3 0 3 0 3 0 1 0 5 0 5 0 5 0 5 0 5 0 3 0 
[ ]
[ 1 ]
[ 2 2 ]
[ 3 3 3 ]
[ 4 4 4 4 ]
```

This example file shows how to nest multiple _for-expressions_ to generate a complex list. `v2`'s initializer uses nested loops plus a filter to select the odd elements of the input, then emit that value that many times into the list. Note the two expansion tokens: the first belongs to the inner loop, and the second to the outer loop.

In `v3` the _take-clause_ `| 10` limits the length of the list. As soon as that length is hit, list comprehension is complete and both loops immediately exit. The _take-clause_ is a _modifier_ for list comprehension. The result object of list comprehension is `std::vector`, and that type has few overloaded operators, so the compiler reserves `operator |` to specify the max list length. There is plenty of symbol space remaining for repetitions, set operations, and the like.

[**locus2.cxx**](locus2.cxx)
```cpp
#include <vector>
#include <cstdio>

int main() {
  
  // Create a vector of vectors, but use different expansion loci.
  // auto m1 = [ [ @range(5)... ] ];
  auto m1 = [ [ for i: 5 => i ... ] ];

  // auto m2 = [ [ @range(5) ]... ];
  auto m2 = [ for i : 5 => [ i ]... ];

  printf("m1:\n[\n");
  for(auto& v : m1) {
    printf("  [ "); printf("%d ", v[:])...; printf("]\n");
  }
  printf("]\n\n");

  printf("m2:\n[\n");
  for(auto& v : m2) {
    printf("  [ "); printf("%d ", v[:])...; printf("]\n");
  }
  printf("]\n");
}
```
```
$ circle locus2.cxx 
$ ./locus2
m1:
[
  [ 0 1 2 3 4 ]
]

m2:
[
  [ 0 ]
  [ 1 ]
  [ 2 ]
  [ 3 ]
  [ 4 ]
]
```

The _for-expression_ prompts its own expansion locus for the expression in the body. To help nail down syntax, you are obligated to expand the _for-expression_ at the same syntactic level in which its written, and not in an enclosing level. This does not reduce the expressiveness of the construct. This example use _for-expressions_ in list comprehension with two different expansion loci: the first puts all five elements in the inner vector; the second creates five vectors with one element each. Since list comprehension is an ordinary expression yielding a prvalue vector, it can be used as the body of the latter _for-expression_, so that each iteration of the loop yields a single-element vector.

### c. Functional fold expressions

C++17 added fold expressions, which are so limited as to be nearly useless. Circle tremendously improves these, turning them into general-purpose reducers of dynamic data. Additionally, the syntax has been expanded to support not just binary operators, but any two-parameter function.

C++ supports four kinds of fold operators on template parameter packs: unary left, unary right, binary left and binary right. The right-associative folds are not supported in Circle on dynamic packs, as right associativity requires right-to-left visitation of data, which is extra confusing for slice data that already has a direction built in. If you want to visit data in right-to-left order, use a left-associative fold with a negative slice step.

```
dynamic left unary fold:
  (... op dynamic-pack-expression [; default-init]) or
  (... function dynamic-pack-expression [; default-init])

dynamic left binary fold:
  (init-value op ... op dynamic-pack-expression) or
  (init-value function ... dynamic-pack-expression)
```

When using a binary functional fold, specify the function expression just once:
use `(INT_MIN std::max ... data[:])` rather than `(INT_MIN std::max ... std::max data[:])`.

As with list comprehensions, the pack expression may either be a slice/range expression or a _for-expression_.

[**fold.cxx**](fold.cxx)
```cpp
#include <vector>
#include <algorithm>
#include <cstdio>

inline int fact(int x) {
  // Use a fold expression to compute factorials. This evaluates the product
  // of integers from 1 to x, inclusive.
  return (... * @range(1:x+1));
}

int main() {
  std::vector<int> v { 4, 2, 2, 2, 5, 1, 1, 9, 8, 7, 1, 7, 4, 1 };
  
  // (... || pack) is a short-circuit fold on operator||.
  bool has_five = (... || (5 == v[:]));
  printf("has_five = %s\n", has_five ? "true" : "false");

  bool has_three = (... || (3 == v[:]));
  printf("has_three = %s\n", has_three ? "true" : "false");

  // Reduce the number of 1s.
  int num_ones = (... + (int)(1 == v[:]));
  printf("has %d ones\n", num_ones);

  // Find the max element using qualified lookup for std::max.
  int max_element = (... std::max v[:]);
  printf("max element = %d\n", max_element);

  // Find the min element using the ADL trick. This uses unqualified lookup
  // for min.
  using std::min;
  int min_element = (... min v[:]);
  printf("min element = %d\n", min_element);

  // Find the biggest difference between consecutive elements.
  int max_diff = (... std::max (abs(v[:] - v[1:])));
  printf("max difference = %d\n", max_diff);

  // Compute the Taylor series for sign. s is the current index, so
  // pow(-1, s) alternates between +1 and -1.
  // The if clause in the for-expression filters out the even elements, 
  // where are zero for sine, and leaves the odd powers. This compacts the
  // vector to 5 elements out of 10 terms.
  int terms = 10;
  std::vector series = [for i : terms if 1 & i => pow(-1, i/2) / fact(i)...];
  printf("series:\n");
  printf("  %f\n", series[:])...;

  // Compute x raised to each odd power. Use @range to generate all odd 
  // integers from 1 to terms, and raise x by that.
  double x = .3;
  std::vector powers = [pow(x, @range(1:terms:2))...];
  printf("powers:\n");
  printf("  %f\n", powers[:])...;

  // Evaluate the series to approximate sine. This is a simple dot
  // product between the coefficient and the powers vectors.
  double sinx = (... + (series[:] * powers[:]));
  printf("sin(%f) == %f\n", x, sinx);
}
```
```
$ circle fold.cxx
$ ./fold
has_five = true
has_three = false
has 4 ones
max element = 9
min element = 1
max difference = 8
series:
  1.000000
  -0.166667
  0.008333
  -0.000198
  0.000003
powers:
  0.300000
  0.027000
  0.002430
  0.000219
  0.000020
sin(0.300000) == 0.295520
```

Fold expressions are a general-purpose reducer. They boil a collection of elements into a single item by iteratively applying the specified operator or function. `(... || (5 == v[:]))` compares `5 == x` for each element `x` in `v`. As soon as one of these tests returns true, the fold expressions returns with a true result object. `||` and `&&`, unless matched with overloaded operators, are short-circuiting operators even in fold expressions, so they'll return as soon as the first true (for `||`) or false (for `&&`) is encountered. Testing if a value exists in an array isn't someone's first idea of "reduction," but it is a reduction on bools, and is cleanly represented using folds on dynamic packs.

`(... std::max v[:])` returns the max element in `v`. The ability to specify functions and not just operators in _fold-expressions_ is new in Circle. If an identifier is provided in the operator spot, argument-dependent lookup is used to find the function in the namespaces associated with the pack's type on the right.

The final sample pulls together _for-expressions_, ranges and extended slices to approximate `sin(x)` from its Taylor series.

```cpp
  int terms = 10;
  std::vector series = [for i : terms if 1 & i => pow(-1, i/2) / fact(i)...];
```

The Taylor series for sine around 0 is the sum of x^i / i!, but only the odd elements, and with alternating signs. The even elements belong to the cosine series. This list comprehension steps from i = 0 to 9, and throws away the even iterations, because they don't belong to this odd function. It computes the alternating signs using `pow(-1, i/2)` (since i is always odd, dividing it by 2 generates alternating even and odd values) and divides that by the result of a factorial call. Because of the filter, the result sequence only contains the even terms.

```cpp
  double x = .3;
  std::vector powers = [pow(x, @range(1:terms:2))...];
```

The next step is to compute x^i given a runtime variable x, for each odd power. The `@range` operator generates a dynamic pack with elements 1, 3, 5, 7, and 9. These are run through the `pow` function, to get us a list of odd powers.

```cpp
  double sinx = (... + (series[:] * powers[:]));
```

The fold expression is a simple inner product with between the odd constants in `series` and the odd variables in `powers`. The terms are multiplied element-wise from their slice expressions, then reduced using additive fold. Better performance would be achieved by inlining the `pow` call from the preceding list comprehension directly into the fold expression (where `powers[:]` is now), but I split the operations in two for clarity.

### d. For-range-initializers

One surprising consumer of dynamic pack expressions (just those resulting from range/slice operators, not _for-expressions_) is range-based _for-statements_ and _for-expressions_. This usage binds the loop's declaration to the dynamic pack's result object at each step. It allows the user to _drain_ a container in an order specified by the extended slice notation. This patches a deficiency in Standard C++, where ranged _for-statements_ aren't capable of anything other than a complete left-to-right visitation of the container. Slices make it easy to start somewhere specific, end somewhere specific, skip elements or run backwards.

[**range_for.cxx**](range_for.cxx)
```cpp
#include <vector>
#include <cstdio>

int main() {
  // Loop over all odd indices and break when i > 10.
  for(int i : @range(1::2)...) {
    printf("%d ", i);
    if(i > 10)
      break;
  }
  printf("\n");
  
  // The same as above, but put the end index in the range.
  for(int i : @range(1:10:2)...)
    printf("%d ", i);
  printf("\n");

  int items[] { 5, 2, 2, 3, 1, 0, 9, 8 };

  // Loop over all but the first item.
  for(int i : items[1:]...)
    printf("%d ", i);
  printf("\n");

  // Loop over items in reverse order.
  for(int i : items[::-1]...)
    printf("%d ", i);
  printf("\n");

  // Bind to the range expression which adds consecutive elements.
  // The items array has 8 elements, but this loop runs through 7 elements,
  // because the slice expression items[1:] starts at index 1 (so only has
  // 7 elements).
  for(int x : items[:] + items[1:]...)
    printf("%d ", x);
  printf("\n");
}
```
```
$ circle range_for.cxx 
$ ./range_for
1 3 5 7 9 11 
1 3 5 7 9 
2 2 3 1 0 9 8 
8 9 0 1 3 2 2 5 
7 4 5 4 1 9 17 
```

Slices add crucial indexing capability to ranged for loops. _for-expression_ also binds to these _for-range_initializers_. What's interesting is that we can bind not only to the lvalue returned by the slice expression, but to the the result object of any expression at all. The final example binds to the sum of consecutive elements from an array, which is a `prvalue int` result object. This prvalue initializes that loop's declaration `x`, and it's the loop machinery associated with the dynamic pack expansion that drives the loop, not the loop machinery of the traditional ranged _for-statement_.

## 3. Modifiers

Dynamic packs benefit from syntax that modifies an expansion or comprehension. The _take-clause_ in list comprehension is one simple example--it dynamically cuts off construction of a list after a certain length is reached. As new challenges arise, additional modifiers will be introduced to help achieve one's goals while mitigating pain of programming.

### a. Sequences

One of the complaints against C++ ranges is that extreme cleverness is required to realize even simple goals. This [detailed blog post](https://www.fluentcpp.com/2019/09/13/the-surprising-limitations-of-c-ranges-beyond-trivial-use-cases/) recounts the author's quest to use ranges to intersperse a scalar between each element in a collection. That is, transform a collection { x0, x1, x2, x3 } into a collection { x0, q, x1, q, x2, q, x3 }. There's a really intricate discussion on this page about the internals and interfacial requirements of range-v3, which necessarily districts from allowing the programmer to simply think about a solution.

The Circle dynamic pack _sequence_ is a syntactic construct that resembles an initializer list: it's a brace pair with a list of one or expressions. But it must be stated inside a list comprehension or in the body of a _for-expression_. In these contexts, it is _not_ a braced initializer for uniform initialization. The actual type of the `std::vector` that the list comprehension returns is only discovered via array type deduction, after each list comprehension element is defined. Therefore, list comprehension doesn't support braced initializers, and this syntax is interpreted instead as a sequence of elements to be inserted. In short, it groups together multiple elements to emit into a list comprehension.

[**sequences.cxx**](sequences.cxx)
```cpp
#include <vector>
#include <string>
#include <cstdio>

int main() {
  std::string s1 = "Hello world";
  std::string s2 = "What's new?";

  // Emit pairs of elements.

  // Print 'H*e*l*l*o* *w*o*r*l*d*'.
  std::string s3 = [{s1[:], '*'}...];
  puts(s3.c_str());

  // Print 'H!e!l!l!o! !w!o!r!l!d!'.
  std::string s4 = [for c : s1 => {c, '!'}...];
  puts(s4.c_str());

  // Print each character with upper the lower case.
  // Print 'HhEeLlLlOo  WwOoRrLlDd'
  std::string s5 = [for c : s1 => { (char)toupper(c), (char)tolower(c) }...];
  puts(s5.c_str());

  // Intersperse elements from s1 with a constant.
  // Print 'H-e-l-l-o- -w-o-r-l-d'
  std::string s6 = [{s1[:-2], '-'}..., s1.back()];
  puts(s6.c_str());

  // Print 'H+e+l+l+o+ +w+o+r+l+d'
  std::string s7 = [for c : s1[:-2]... => { c, '+'}..., s1.back()];
  puts(s7.c_str());

  // Interleave two strings.
  // Print 'HWehlalto' sw onrelwd?'
  std::string s8 = [{s1[:], s2[:]}...];
  puts(s8.c_str());
}
```
```
$ circle sequences.cxx 
$ ./sequences
H*e*l*l*o* *w*o*r*l*d*
H!e!l!l!o! !w!o!r!l!d!
HhEeLlLlOo  WwOoRrLlDd
H-e-l-l-o- -w-o-r-l-d
H+e+l+l+o+ +w+o+r+l+d
HWehlalto' sw onrelwd?
```

## Static slice expressions

After implementing slice expressions yielding dynamic types, I backported the slice operator to yield static packs when applied to template parameter packs or objects, members or parameters with array or tuple-like types.

### Static slices on template parameter packs

[**slice2.cxx**](slice2.cxx)
```cpp
#include <iostream>

template<int... x, typename... types_t>
void func1(types_t... args) {
  std::cout<< "Expansion expression of static parameter pack:\n";

  std::cout<< "  Non-type template parameters:\n";
  std::cout<< "    "<< x<<"\n" ...;

  std::cout<< "  Type template parameters:\n";
  std::cout<< "    "<< @type_string(types_t)<< "\n" ...;

  std::cout<< "  Function template parameters:\n";
  std::cout<< "    "<< args<< "\n" ...;
}

template<int... x, typename... types_t>
void func2(types_t... args) {
  std::cout<< "\nReverse-order direct pack indexing with ...[index]:\n";

  std::cout<< "  Non-type template parameters:\n";
  @meta for(int i = sizeof...(x) - 1; i >= 0; --i)
    std::cout<< "    "<< x...[i]<<"\n";

  std::cout<< "  Type template parameters:\n";
  @meta for(int i = sizeof...(x) - 1; i >= 0; --i)
    std::cout<< "    "<< @type_string(types_t...[i])<< "\n";

  std::cout<< "  Function template parameters:\n";
  @meta for(int i = sizeof...(x) - 1; i >= 0; --i)
    std::cout<< "    "<< args...[i]<< "\n";
}

template<int... x, typename... types_t>
void func3(types_t... args) {
  std::cout<< "\nReverse-order pack slices with ...[begin:end:step]:\n";

  std::cout<< "  Non-type template parameters:\n";
  std::cout<< "    "<< x...[::-1]<<"\n" ...;

  std::cout<< "  Type template parameters:\n";
  std::cout<< "    "<< @type_string(types_t...[::-1])<< "\n" ...;

  std::cout<< "  Function template parameters:\n";
  std::cout<< "    "<< args...[::-1]<< "\n" ...;
}

int main() {
  func1<100, 200, 300>(4, 5l, 6ll);
  func2<100, 200, 300>(4, 5l, 6ll);
  func3<100, 200, 300>(4, 5l, 6ll);
}
```
```
$ circle slice2.cxx 
$ ./slice2
Expansion expression of static parameter pack:
  Non-type template parameters:
    100
    200
    300
  Type template parameters:
    int
    long
    long long
  Function template parameters:
    4
    5
    6

Reverse-order direct pack indexing with ...[index]:
  Non-type template parameters:
    300
    200
    100
  Type template parameters:
    long long
    long
    int
  Function template parameters:
    6
    5
    4

Reverse-order pack slices with ...[begin:end:step]:
  Non-type template parameters:
    300
    200
    100
  Type template parameters:
    long long
    long
    int
  Function template parameters:
    6
    5
    4
```

Circle adds capability to expand static pack expressions with `...` at the end of an expression statement. Any expression built around a reference to a template parameter or variadic function template parameter inherits that static pack bit, and carries it through until the expansion locus. This triggers template substitution, and each element is instantiated in a loop. `func1` uses an expansion expression to print the contents of a non-type parameter pack, a type parameter pack (formatting the types as strings) and variadic function template parameters.

From its earliest incarnations Circle has included a `...[index]` operator for subscripting template parameter packs. The `@meta` _for-statement_ generates compile-time indices for iteratively indexing packs.  `func2` loops in reverse order, plucking out each pack element and printing it.

`func3` is the best of both approaches. It is concise, because it uses an expansion expression instead of a for loop to visit the pack. It is flexible, because it uses the static slice operator `...[begin:end:step]` in order to specify reverse-order visitation. The static slice operator takes a static pack expression and yields a static pack expression. The dynamic slice operator, by contrast, takes a non-pack expression and turns it into dynamic pack by coordinating the generation of a runtime loop over the pack's elements. The static slice operator doesn't generate a loop, but rather maps indices from a slice space to a container space, using the begin, end and step indices.

[**pack_decl.cxx**](pack_decl.cxx)
```cpp
#include <cstdio>

template<typename type_t>
void print_structure() {
  printf("%s:\n", @type_string(type_t));
  printf("  %s\n", @member_decl_strings(type_t))...;
}

template<typename... types_t>
struct tuple_t {
  types_t @(int...)...;
};

template<typename... types_t>
struct reverse_tuple_t {
  types_t...[::-1] @(int...)...;
};

@meta print_structure<tuple_t<int, char, double*> >();
@meta print_structure<reverse_tuple_t<int, char, double*> >();

int main() { }
```
```
$ circle pack_decl.cxx 
tuple_t<int, char, double*>:
  int _0
  char _1
  double* _2
reverse_tuple_t<int, char, double*>:
  double* _0
  char _1
  int _2
```

Circle supports static expansion loci after both object and data member declarations. The type in the declaration may be a pack or non-pack type, but the name of the declaration must be a pack of dynamic name identifiers. Finishing the declaration statement with the expansion token `...` causes one instantiation per pack element to be created during template instantiation. `int...` is special value-dependent expression which yields the index of the current pack element during substitution. The dynamic name operator `@()` converts this pack index into an underscore-prefixed identifier at each step.

The static slice operator `...[begin:end:step]` transforms the template parameter pack `types_t` in `reverse_tuple_t`. The size of the pack remains the same, but substitution through the slice reverses the order in which the pack elements are accessed.

### Static slices on tuple-like objects

The static slice operator `...[begin:end:step]` also works when applied to non-pack expressions with array, tuple-like or class types. It presents busts up the entity and exposes it as a _heterogeneous_ non-type static pack. The same semantic rules for fixing structured bindings to initializers is at play here:

1. If the operand is an array, each element in the static parameter pack is one array element.
1. If specializing `std::tuple_size` on the operand's type finds a partial template specialization (or technically a specialization that's not incomplete), the object is treated as "tuple-like." `std::tuple_element` breaks the object apart into elements. `std::array`, `std::pair` and `std::tuple` all provide specializations for these class templates.
1. If the operand is another class type, the non-static public data members are exposed as a static parameter pack.

In each case, only the size of the pack (or the size of the container) must be known at compile time. The data values may be known at compile time (either being meta or constexpr), or may be only known at runtime.

[**slice3.cxx**](slice3.cxx)
```cpp
#include <utility>
#include <tuple>
#include <iostream>

// Print comma-separated arguments.
template<typename... types_t>
void print_args(types_t... args) {
  std::cout<< "[ ";
  
  // Use static slicing to print all but the last element, followed by
  // a comma.
  std::cout<< args...[:-2]<< ", "...;

  // Use direct indexing to print the last element, if there is one.
  // The static subscript ...[-1] refers to the last element.
  // ...[-2] refers to the second-to-last element, and so on.
  if constexpr(sizeof...(args) > 0)
    std::cout<< args...[-1]<< " ";

  std::cout<< "]\n";
}

template<int... values>
void print_nontype() {
  std::cout<< "[ ";
  
  // Print template non-type arguments the same way you print function args.
  std::cout<< values...[:-2]<< ", "...;
  
  // Print the last non-type argument.
  if constexpr(sizeof...(values) > 0)
    std::cout<< values...[-1]<< " ";

  std::cout<< "]\n";
}

int main() {
  auto tuple = std::make_tuple('A', 1, 2.222, "Three");

  // Use static indexing to turn a tuple into a parameter pack. Expand it
  // into function arguments.
  std::cout<< "tuple to pack in forward order:\n";
  print_args(tuple...[:] ...);

  // Or expand it in reverse order.
  std::cout<< "\ntuple to pack in reverse order:\n";
  print_args(tuple...[::-1] ...);

  // Or send the even then the odd elements.
  std::cout<< "\neven then odd tuple elements:\n";
  print_args(tuple...[0::2] ..., tuple...[1::2] ...);

  // Pass indices manually to a template.
  std::cout<< "\ntemplate non-type arguments sent the old way:\n";
  print_nontype<3, 4, 5, 6>();

  // Or use static slicing to turn an array, class or tuple-like object
  // into a parameter pack and expand that into a template-arguments-list.
  std::cout<< "\ntemplate non-type arguments expanded from an array:\n";
  constexpr int values[] { 7, 8, 9, 10 };
  print_nontype<values...[:] ...>();
}
```
```
$ circle slice3.cxx && ./slice3
tuple to pack in forward order:
[ A, 1, 2.222, Three ]

tuple to pack in reverse order:
[ Three, 2.222, 1, A ]

even then odd tuple elements:
[ A, 2.222, 1, Three ]

template non-type arguments sent the old way:
[ 3, 4, 5, 6 ]

template non-type arguments expanded from an array:
[ 7, 8, 9, 10 ]
```

This is a pretty extraordinary example. The slice operator is used to expose an `std::tuple` as a static parameter pack and to re-arrange its elements when expanded into a function argument list. That function, `print_args` prints all elements in the variadic function parameter by using another static slice operator. 

The function comma-separates elements by specifying the slice arguments `[:-2]`, which prints all elements from the first (the implicit 0 begin index) to -2, indicates two elements from the end of the variadic argument set. That is, it prints all elements except the last. An `if constexpr` clause prints the last element without a paired comma, if the argument has more than zero elements. The `...[index]` static subscript operator decodes signed indices. If the index is negative, the size of the pack is added. Therefore, -1 refers to size - 1, which is the index for the last element.

[**slice4.cxx**](slice4.cxx)
```cpp
#include <iostream>

template<typename... types_t>
struct tuple_t {
  types_t @(int...) ...;
};

template<typename... types_t>
tuple_t(types_t... args) -> tuple_t<types_t...>;

template<typename type_t>
void print_object(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int... << ") "<< 
    @type_string(decltype(obj...[:]))<< " : "<< 
    obj...[:]<< "\n" ...;
}

template<typename type_t>
void print_reverse(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int... << ") "<< 
    @type_string(decltype(obj...[::-1]))<< " : "<< 
    obj...[::-1]<< "\n" ...;
}

template<typename type_t>
void print_odds(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int... << ") "<< 
    @type_string(decltype(obj...[1::2]))<< " : "<< 
    obj...[1::2]<< "\n" ...;
}

int main() {
  tuple_t obj { 3.14, 100l, "Hi there", "Member 3", 'q', 19u };

  std::cout<< "print_object:\n";
  print_object(obj);

  std::cout<< "\nprint_reverse:\n";
  print_reverse(obj);

  std::cout<< "\nprint_odds:\n";
  print_odds(obj);
}

```
```
$ circle slice4.cxx && ./slice4
print_object:
tuple_t<double, long, const char*, const char*, char, unsigned>
  0) double : 3.14
  1) long : 100
  2) const char* : Hi there
  3) const char* : Member 3
  4) char : q
  5) unsigned : 19

print_reverse:
tuple_t<double, long, const char*, const char*, char, unsigned>
  0) unsigned : 19
  1) char : q
  2) const char* : Member 3
  3) const char* : Hi there
  4) long : 100
  5) double : 3.14

print_odds:
tuple_t<double, long, const char*, const char*, char, unsigned>
  0) long : 100
  1) const char* : Member 3
  2) unsigned : 19
```

This example uses static slices to break tuple-like objects into packs and prints the expanded contents with annotations. During static pack expansion, the `int...` operator yields the current index of expansion as an integer. (It's a value-dependent expression.) `@type_string(decltype(obj...[:]))` renders the type of the current slice expansion into a character array. This may be different from the Circle intrinsic `@member_type_strings`. The former expression treats array types, `std::array`s, `std::pair`s and `std::tuple`s by running their contents through `std::tuple_element`. The `@member_type_strings` operator returns character arrays spelling out the type of each non-static data member of the argument type.

[**slice5.cxx**](slice5.cxx)
```cpp
#include <vector>
#include <array>
#include <iostream>

template<typename type_t>
void print_dynamic(const type_t& obj) {
  std::cout<< "[ ";

  // A homogeneous print operation that uses dynamic pack expansion
  // and generates a runtime loop. The container must implement .begin()
  // and .end().
  std::cout<< obj[:]<< " "...;

  std::cout<< "]\n";
}

template<typename type_t>
void print_static(const type_t& obj) {
  std::cout<< "[ ";

  // A heterogenous print operation. Uses static pack expansion. Works on
  // member objects of class types, regular arrays, plus types implementing
  // std::tuple_size, such as std::array, std::pair and std::tuple.
  std::cout<< obj...[:]<< " "...;

  std::cout<< "]\n";
}

int main() {
  std::array<int, 8> array { 1, 2, 3, 4, 5, 6, 7, 8 };

  // Dynamic pack indexing generates a loop.
  array[:] *= 2 ...;
  print_dynamic(array);

  // Static pack indexing performs template substitution to unroll
  // the operation.
  ++array...[:] ...;
  print_static(array);

  // Use list comprehension to generate an std::vector.
  // Expansion of dynamic slice creates a dynamic loop.
  std::vector v1 = [ array[:] * 3... ];
  print_dynamic(v1);

  // Expansion of static slice expansion occurs during substitution.
  // This supports using heterogeneous containers as initializers in
  // list comprehension and uniform initializers.
  std::vector v2 = [ array...[:] * 4... ];
  print_dynamic(v2);

  // Use static slice expansion to create an initializer list for a 
  // builtin array. This won't work with the dynamic slice operator [:], 
  // because the braced initializer must have a compile-time set number of 
  // elements.
  int array2[] { array...[:] * 5 ... };
  print_static(array2);

  // Create a braced initializer in forward then reverse order.
  int forward_reverse[] { array...[:] ..., array...[::-1] ...};
  print_static(forward_reverse);

  // Create a braced initializer with evens then odds.
  int parity[] { array...[0::2] ..., array...[1::2] ... };
  print_static(parity);

  // Use a compile-time loop to add up all elements of array3.
  int static_sum = (... + array...[:]);
  printf("static sum = %d\n", static_sum);

  // Use a dynamic loop to add up all elements of array3.
  int dynamic_sum = (... + array[:]);
  printf("dynamic sum = %d\n", dynamic_sum);
}

```
```
$ circle slice5.cxx && ./slice5
[ 2 4 6 8 10 12 14 16 ]
[ 3 5 7 9 11 13 15 17 ]
[ 9 15 21 27 33 39 45 51 ]
[ 12 20 28 36 44 52 60 68 ]
[ 15 25 35 45 55 65 75 85 ]
[ 3 5 7 9 11 13 15 17 17 15 13 11 9 7 5 3 ]
[ 3 7 11 15 5 9 13 17 ]
static sum = 80
dynamic sum = 80
```

This example shows the similar and contrasting aspects of dynamic pack and static pack expansion. Static packs may only be used on containers where the pack size is part of the type, such as builtin arrays, `std::array`, `std::pair`, `std::tuple` and other class objects, in which the non-static data members are collected.

`std::tuple` is compatible with the static slice operator, but not the dynamic slice operator, because it doesn't implement `.begin()` and `.end()` accessors. `std::vector` is compatible with the dynamic slice operator, but not the static slice operator, because it doesn't implement a partial template of `std::tuple_size`. `std::array` is compatible with both, because it provides both the accessor member functions and a `std::tuple_size` partial template definition. Builtin arrays are also compatible with both, but as non-class types are handled specially by the compiler.

Both static and dynamic packs (generated by static or dynamic slice operators) can be expanded inside list comprehension. This list comprehension yields a prvalue `std::vector` result object, but may also decay and serve as the backing store for an `std::initializer_list` constructor or assignment. 

Only static packs may be expanded inside braced initializers. Unlike list comprehensions, the backing store for braced initializer-generated `std::initializer_list`s is allocated statically, so the dynamic packs aren't supported.

Both dynamic and static packs may be expanded inside functional fold expressions. In essence, the static slice version is guaranteed to be loop unrolled by the compiler frontend. The dynamic slice version corresponds to a dynamic loop emitted as LLVM IR by the frontend, but what machine code is generated from that is up to the backend.

It's important to keep in mind that static slices support _heterogeneous_ packs, while dynamic slices only support _homogeneous_ packs. All STL containers with `.begin` and `.end` accessors hold _homogeneous_ data, which is usually the type that the class template is specialized over. STL containers holding parameterized _heterogeneous_ data usually specify their contained data types with variadic class template parameters, and expose access to the contents with `std::tuple_element` partial templates.

The Circle slice operators help abstract these differences, by providing similar-looking operators that are compatible with the same pack consumers.

## Circle vs C++ ranges

There is lots of overlap between the problems addressed by Circle's dynamic packs and C++ 20's [ranges](https://en.cppreference.com/w/cpp/ranges)/[range-v3](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html). Ranges doesn't requires any compiler modifications, and is entirely a library-based solution. 

Ranges goes through huge effort to simulate lazy evaluation. As with older STL algorithms like `std::for_each`, the user is required to capture their program's data dependencies into a closure, pass that closure to a utility function in which some loops are written, then wait for the closure function to be invoked at each step in the algorithm. As the user's task becomes more complicated and its data dependencies grow, the capture cost increases, and more strain is put on the compiler backend to inline away multiple calls and captures.

With Circle dynamic packs, there is no closure and no function call. Collection operations are a first-class part of the language. The only library dependency is `<vector>`, and only when list comprehension is used.

### [Hello, Ranges!](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-hello)

```cpp
#include <iostream>
#include <range/v3/all.hpp> // get everything
#include <string>
using std::cout;
int
main()
{
    std::string s{"hello"};
    // output: h e l l o
    ranges::for_each(s, [](char c) { cout << c << ' '; });
    cout << '\n';
}
```

**With Circle**

[**hello.cxx**](hello.cxx)
```cpp
#include <iostream>
#include <string>

int main() {
  std::string s = "hello";
  std::cout<< s[:]<< ' ' ...;   // Prints 'h e l l o '
  std::cout<< '\n';
}
```

### [any_of, all_of, none_of](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-any-all-none)

```cpp
// Demonstrates any_of, all_of, none_of
// output
// vector: [6,2,3,4,5,6]
// vector any_of is_six: true
// vector all_of is_six: false
// vector none_of is_six: false
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/algorithm/none_of.hpp>
#include <range/v3/view/all.hpp>
#include <iostream>
#include <vector>
using std::cout;
auto is_six = [](int i) { return i == 6; };
int
main()
{
    std::vector<int> v{6, 2, 3, 4, 5, 6};
    cout << std::boolalpha;
    cout << "vector: " << ranges::views::all(v) << '\n';
    cout << "vector any_of is_six: " << ranges::any_of(v, is_six) << '\n';
    cout << "vector all_of is_six: " << ranges::all_of(v, is_six) << '\n';
    cout << "vector none_of is_six: " << ranges::none_of(v, is_six) << '\n';
}
```

**With Circle**

[**any_of.cxx**](any_of.cxx)
```cpp
#include <vector>
#include <iostream>

int main() {
  using std::cout;

  std::vector<int> v { 6, 2, 3, 4, 5, 6 };

  cout<< std::boolalpha;
  cout<< "vector any_of is 6: "<< (... || (6 == v[:]))<< '\n';
  cout<< "vector all_of is 6: "<< (... && (6 == v[:]))<< '\n';
  cout<< "vector none_of is 6: "<< (... && (6 != v[:]))<< '\n';
}
```

### [count](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-count)

```cpp
// This example demonstrates counting the number of
// elements that match a given value.
// output...
// vector:   2
// array:    2
#include <iostream>
#include <range/v3/algorithm/count.hpp> // specific includes
#include <vector>
using std::cout;
int
main()
{
    std::vector<int> v{6, 2, 3, 4, 5, 6};
    // note the count return is a numeric type
    // like int or long -- auto below make sure
    // it matches the implementation
    auto c = ranges::count(v, 6);
    cout << "vector:   " << c << '\n';

    std::array<int, 6> a{6, 2, 3, 4, 5, 6};
    c = ranges::count(a, 6);
    cout << "array:    " << c << '\n';
}
```

**With Circle**

[**count.cxx**](count.cxx)
```cpp
#include <vector>
#include <array>
#include <iostream>

int main() {
  using std::cout;

  // Count the number of 6s.
  // Promote each comparison to int, because adding bools in a dynamic
  // fold expression will return a bool type.
  std::vector<int> v { 6, 2, 3, 4, 5, 6 };
  int count1 = (... + (int)(6 == v[:]));
  cout<< "vector: "<< count1<< '\n';

  // Do it with an array.
  std::array<int, 6> a { 6, 2, 3, 4, 5, 6 };
  int count2 = (...+ (int)(6 == a[:]));
  cout<< "array: "<< count2<< '\n';
}
```

### [count_if](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-count_if)

```cpp
// This example counts element of a range that match a supplied predicate.
// output
// vector:   2
// array:    2
#include <array>
#include <iostream>
#include <range/v3/algorithm/count_if.hpp> // specific includes
#include <vector>

using std::cout;
auto is_six = [](int i) -> bool { return i == 6; };

int main()
{
    std::vector<int> v{6, 2, 3, 4, 5, 6};
    auto c = ranges::count_if(v, is_six);
    cout << "vector:   " << c << '\n'; // 2

    std::array<int, 6> a{6, 2, 3, 4, 5, 6};
    c = ranges::count_if(a, is_six);
    cout << "array:    " << c << '\n'; // 2
}
```

**With Circle**

[**count_if.cxx**](count_if.cxx)
```cpp
#include <vector>
#include <array>
#include <iostream>

int main() {
  using std::cout;

  // This is the identical implementation as count.cxx. 
  // ranges::count and ranges::count_if are implemented with the same
  // fold expression in Circle.
  std::vector<int> v { 6, 2, 3, 4, 5, 6 };
  int count1 = (... + (int)(6 == v[:]));
  cout<< "vector: "<< count1<< '\n';

  std::array<int, 6> a { 6, 2, 3, 4, 5, 6 };
  int count2 = (...+ (int)(6 == a[:]));
  cout<< "array: "<< count2<< '\n';
}
```

### [for_each on sequence containers](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-for_each-seq)

```cpp
// Use the for_each to print from various containers
// output
// vector:   1 2 3 4 5 6
// array:    1 2 3 4 5 6
// list:     1 2 3 4 5 6
// fwd_list: 1 2 3 4 5 6
// deque:    1 2 3 4 5 6
#include <array>
#include <deque>
#include <forward_list>
#include <iostream>
#include <list>
#include <queue>
#include <range/v3/algorithm/for_each.hpp> // specific includes
#include <stack>
#include <vector>

using std::cout;

auto print = [](int i) { cout << i << ' '; };

int main()
{
    cout << "vector:   ";
    std::vector<int> v{1, 2, 3, 4, 5, 6};
    ranges::for_each(v, print); // 1 2 3 4 5 6

    cout << "\narray:    ";
    std::array<int, 6> a{1, 2, 3, 4, 5, 6};
    ranges::for_each(a, print);

    cout << "\nlist:     ";
    std::list<int> ll{1, 2, 3, 4, 5, 6};
    ranges::for_each(ll, print);

    cout << "\nfwd_list: ";
    std::forward_list<int> fl{1, 2, 3, 4, 5, 6};
    ranges::for_each(fl, print);

    cout << "\ndeque:    ";
    std::deque<int> d{1, 2, 3, 4, 5, 6};
    ranges::for_each(d, print);
    cout << '\n';
}
```

**With Circle**

[**for_each.cxx**](for_each.cxx)
```cpp
#include <vector>
#include <array>
#include <list>
#include <forward_list>
#include <queue>
#include <iostream>

int main() {
  using std::cout;

  cout<< "vector:   ";
  std::vector<int> v { 1, 2, 3, 4, 5, 6 };
  cout<< v[:]<< ' ' ...;

  cout<< "\narray:    ";
  std::array<int, 6> a { 1, 2, 3, 4, 5, 6 };
  cout<< a[:]<< ' ' ...;

  cout<< "\nlist:     ";
  std::list<int> ll { 1, 2, 3, 4, 5, 6 };
  cout<< ll[:]<< ' ' ...;

  cout<< "\nfwd_list: ";
  std::forward_list<int> fl { 1, 2, 3, 4, 5, 6 };
  cout<< fl[:]<< ' ' ...;

  cout<< "\ndeque:    ";
  std::deque<int> d { 1, 2, 3, 4, 5, 6 };
  cout<< d[:]<< ' ' ...;
  cout<< '\n';
}
```
```
$ circle for_each2.cxx
$ ./for_each
vector:   1 2 3 4 5 6 
array:    1 2 3 4 5 6 
list:     1 2 3 4 5 6 
fwd_list: 1 2 3 4 5 6
deque:    1 2 3 4 5 6 
```

It's worth noting that `std::forward_list` is such a lightweight container that it doesn't even implement a `size` member function. Since Circle can't compute a size to control execution of dynamic pack expansion on a `forward_list` slice, it compares the iterator representing slice progress against the container's `end` member function at each step. For a pack expansion over a single slice, there's no performance consequence for this, since the predicate is still only a single comparison. However as more and more size-less slices are involved in an expansion, you'll suffer a predicate test for each of them at each step at runtime.

### [for_each on associative containers](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-for_each-assoc)

```cpp
#include <iostream>
#include <map>
#include <range/v3/algorithm/for_each.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
using std::cout;
using std::string;

auto print = [](int i) { cout << i << ' '; };
// must take a pair for map types
auto printm = [](std::pair<string, int> p) {
    cout << p.first << ":" << p.second << ' ';
};

int
main()
{
    cout << "set:           ";
    std::set<int> si{1, 2, 3, 4, 5, 6};
    ranges::for_each(si, print);

    cout << "\nmap:           ";
    std::map<string, int> msi{{"one", 1}, {"two", 2}, {"three", 3}};
    ranges::for_each(msi, printm);

    cout << "\nunordered map: ";
    std::unordered_map<string, int> umsi{{"one", 1}, {"two", 2}, {"three", 3}};
    ranges::for_each(umsi, printm);

    cout << "\nunordered set: ";
    std::unordered_set<int> usi{1, 2, 3, 4, 5, 6};
    ranges::for_each(usi, print);
    cout << '\n';
}
```

**With Circle**

[**for_each2.cxx**](for_each2.cxx)
```cpp
#include <vector>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

int main() {
  using std::cout;

  cout<< "set:           ";
  std::set<int> si { 1, 2, 3, 4, 5, 6 };
  cout<< si[:]<< ' ' ...;

  cout<< "\nmap:           ";
  std::map<std::string, int> msi {{"one", 1}, {"two", 2}, {"three", 3}};
  cout<< msi[:].first<< ':'<< msi[:].second<< ' ' ...;

  cout<< "\nunordered map: ";
  std::unordered_map<std::string, int> umsi {{"one", 1}, {"two", 2}, {"three", 3}};
  cout<< umsi[:].first<< ':'<< umsi[:].second<< ' ' ...;

  cout<< "\nunordered set: ";
  std::unordered_set<int> usi { 1, 2, 3, 4, 5, 6 };
  cout<< usi[:]<< ' ' ...;
  cout<< '\n';
}
```
```
$ circle for_each2.cxx 
$ ./for_each2
set:           1 2 3 4 5 6 
map:           one:1 three:3 two:2 
unordered map: three:3 two:2 one:1 
unordered set: 6 5 4 3 2 1 
```

### [is_sorted](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-is_sorted)

```cpp
#include <array>
#include <iostream>
#include <range/v3/algorithm/is_sorted.hpp> // specific includes
#include <vector>
using std::cout;

int
main()
{
    cout << std::boolalpha;
    std::vector<int> v{1, 2, 3, 4, 5, 6};
    cout << "vector:   " << ranges::is_sorted(v) << '\n';

    std::array<int, 6> a{6, 2, 3, 4, 5, 6};
    cout << "array:    " << ranges::is_sorted(a) << '\n';
}
```

**With Circle**

[**is_sorted.cxx**](is_sorted.cxx)
```cpp
#include <vector>
#include <array>
#include <iostream>

int main() {
  using std::cout;
  cout<< std::boolalpha;

  std::vector<int> v { 1, 2, 3, 4, 5, 6 };
  bool is_sorted1 = (... && (v[:] <= v[1:]));
  cout<< "vector: "<< is_sorted1<< '\n';

  std::array<int, 6> a { 6, 2, 3, 4, 5, 6 };
  bool is_sorted2 = (... && (a[:] <= a[1:]));
  cout<< "array:  "<< is_sorted2<< '\n';
}

```
```
$ circle is_sorted.cxx 
$ ./is_sorted 
vector: true
array:  false
```

### [Filter and transform](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-filter-transform)

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
using std::cout;

int main()
{
    std::vector<int> const vi{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    using namespace ranges;
    auto rng = vi | views::filter([](int i) { return i % 2 == 0; }) |
               views::transform([](int i) { return std::to_string(i); });
    // prints: [2,4,6,8,10]
    cout << rng << '\n';
}
```

**With Circle**

[**filter.cxx**](filter.cxx)
```cpp
#include <vector>
#include <string>
#include <cstdio>

int main() {
  std::vector<int> vi { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

  // Create a vector of strings. Filter for the even elements and convert 
  // to strings.
  std::vector v2 = [for i : vi if 0==i%2 => std::to_string(i) ...];

  // Print the strings out.
  printf("%s ", v2[:].c_str())...; printf("\n"); // Prints '2 4 6 8 10 '.
}
```

## [Generate ints and accumulate](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-accumulate-ints)

```cpp
#include <iostream>
#include <vector>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
using std::cout;

int main()
{
    using namespace ranges;
    int sum = accumulate(views::ints(1, unreachable) | views::transform([](int i) {
                             return i * i;
                         }) | views::take(10),
                         0);
    // prints: 385
    cout << sum << '\n';
}
```

**With Circle**

[**accumulate.cxx**](accumulate.cxx)
```cpp
// Sums the first ten squares and prints them, using views::ints to generate
// and infinite range of integers, views::transform to square them, views::take
// to drop all but the first 10, and accumulate to sum them.

#include <vector>
#include <iostream>

int sq(int x) { return x * x; }

int main() {
  int sum = (... + sq(@range(1:11)));
  std::cout<< sum<< '\n';   // Prints 385
}
```

### [Convert a range comprehension to a vector](https://ericniebler.github.io/range-v3/md___users_eniebler__code_range-v3_doc_examples.html#example-comprehension-conversion)

```cpp
// Use a range comprehension (views::for_each) to construct a custom range, and
// then convert it to a std::vector.
#include <iostream>
#include <vector>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
using std::cout;

int main()
{
    using namespace ranges;
    auto vi = views::for_each(views::ints(1, 6),
                              [](int i) { return yield_from(views::repeat_n(i, i)); }) |
              to<std::vector>();
    // prints: [1,2,2,3,3,3,4,4,4,4,5,5,5,5,5]
    cout << views::all(vi) << '\n';
}
```

**With Circle**

[**comprehension.cxx**](comprehension.cxx)
```cpp
#include <vector>
#include <cstdio>

int main() {
  std::vector v = [ for i : @range(1:6)... => for i2 : i => i ... ... ];

  // Prints '1 2 2 3 3 3 4 4 4 4 5 5 5 5 5 '.
  printf("%d ", v[:])...; printf("\n");
}
```

## Points of evolution

**Should the slice syntax be an overloadable operator?**

If it were, what kind of result object would it return? In the current treatment, the dynamic pack bit is carried by expressions only, and not by types. There is no type representing a "dynamic pack." Providing `begin`, `end` and `size` member functions allows you to make your own types compatible with Circle slices, but slice overloading is not allowed in the conventional way.

**Should the expansion syntax be an overloadable operator?**

It might be possible to write your own dynamic pack consumers by overloading `...`. Again, this would require pack-aware user-defined types, which are not currently part of the picture.

**Can we separate dynamic pack expansion from static pack expansion?**

`...` was chosen to expand dynamic packs, because it already was used to expand static packs. This may limit flexibility somewhat, in that each expansion must contain only static or dynamic packs, but not both, since we don't have a separate token to distinguish which one should get expanded. Using a `..` token to expand dynamic packs would give us finer-grained control over expansions. Unfortunately most C++ text editors freak out when encountering the `..` token. I didn't think this slight increase in flexibility was worth dealing with crabby editors.

**Can we have pack-aware io?**

Printf and iostreams are not pack aware, so printing a slice expression currently involves an expansion expression statement and optional begin and end non-expanded function calls:

```cpp
printf("[ "); printf("%d ", v[:])...; printf("]\n");
```

The picture is even more complicated if we want to comma-separate the elements.

It's a priority to design a first-class string formatting utility, as a dynamic pack consumer, which leverages reflection to decompose structures and expansion to support dynamic pack expressions.

**Can for-expressions be extended with else and if-else clauses?**

A richer syntax might be something like

`for x : c if pred => a else b`,

which is read "for x in x if pred then a else b." The simplest form is already supported using the condition operator, such as 'for x : c => pred ? a : b', but an explicit if-else form would allow sequences to generate a variable number of list outputs per step.

Allowing another _if_- or _if-else_-clause to follow the _for-expressions_'s _if-else_ clause would give us maximum flexibility, but would compromise legibility. It may be useful to support [patterns](../pattern/pattern.md), and extend them to return sequences which would be streamed into list comprehensions.

**Can we integrate dynamic packs with co-routines?**

Hopefully. I haven't implemented co-routines. As of yet they are pretty underdefined. But they seem to promise a way to save the state of a computation, allow the program to go do something else, then come back and execute a few more iterations from it.
