# Spaceship operator

The most complex feature of C++20's many additions and modifications of the language is the spaceship operator. The token `<=>` represents a "three-way" comparison, which in its builtin form returns one of five structures representing possible states of _less_, _greater_, _equal_, _nonequal_ or _unordered_ (in the case of NaN floating-point values).

All the structured control-flow conditions in C++ take only binary expressions: the argument in an _if-statement_ or ternary expression evaluates to only two states, not three or more. You won't be using the spaceship operator in control-flow constructs. Rather, this language extension is intended to help _generate_ the familiar equivalence (`==` and `!=`) and relational (`<`, `<=`, `>` and `>=`) operators for user-defined types. 

The semantics of the spaceship operator are complicated, and you benefit from the language's ability to check correctness at compile time.

Both builtin and user-defined `<=>` operators return one of five classes:

1. [`weak_equality`](http://eel.is/c++draft/cmp.weakeq)
1. [`strong_equality`](http://eel.is/c++draft/cmp.strongeq)
1. [`partial_ordering`](http://eel.is/c++draft/cmp.partialord)
1. [`weak_ordering`](http://eel.is/c++draft/cmp.weakord)
1. [`strong_ordering`](http://eel.is/c++draft/cmp.strongord)

The "strong" types imply a _transitive_ property: that is, if a == b and b == c, then a == c. Simiarly, it implies _substitutability_: that is, if a == b, then f(a) == f(b).

The builtin `<=>` on scalar types returns these types:
1. For integer and enum types, `strong_ordering`.
1. For floating-point types, `partial_ordering`.
1. For pointers to objects, `strong_ordering`.
1. For pointers to functions, pointers to member objects and `std::nullptr_t`, `strong_equality`.

The result of a floating-point comparison is especially weak, since the presence of a NaN value causes any two-way comparison to evaluate false. When subjected to the three-way comparison, NaN results in an _unordered_ value.

These builtin comparisons supply the building blocks for constructing the three-way comparison for user-defined types. From this operator, implementations of the four relational operators are inferred.

## The default operator<=>

[**spaceship1.cxx**](spaceship1.cxx)
```cpp
#include <cstdio>
#include <compare>

struct int3_t {
  int x, y, z;

  auto operator<=>(const int3_t& rhs) const = default;
};

int main() {
  int3_t a { 1, 2, 3 }, b { 1, 2, 4 }, c { 1, 1, 5 };
  printf("%s\n", @type_name(decltype(a <=> b)));

  bool x = a < b;
  printf("%d\n", x);

  bool y = a < c;
  printf("%d\n", y);

  bool z = b == c;
  printf("%d\n", z);

  return 0;
}
```
```
$ circle spaceship1.cxx
$ ./spaceship1
std::strong_ordering
1
0
0
```

Consider defining a vector of integers. We want these to compare in the expected way: first compare the .x members; if those are equal, compare the .y members; if those are equal, compare the .z members. All six equivalence and relational operators make sense under these comparison rules. The default-generated `operator<=>` will support all of these operators.

To realize this default behavior, just declare a defaulted non-static member function `operator<=>` that returns `auto` and takes a const lvalue reference to its own type. In C++ 20, this causes a cascade of behaviors:

## Defaulted operators

* The default definition of `operator<=>` recursively evaluates `<=>` on each of its direct base class and member subobjects, in declaration order.
* A defaulted `operator<=>` creates an implicit declaration of a defaulted `operator==` with the same function signature.
* The default definition of `operator==` (be it declared implicitly or explicitly) recursively evaluates `==` on each of its direct base class and member subobjects, in declaration order.

The default implementation of either `<=>` or `==` is valid if all subobjects have valid, accessible and non-deleted comparisons. Otherwise, the function is reset to _deleted_ when called (similar to default definitions for copy and move constructors and assignment operators). 

## Rewritten relational operators

* Evaluation of the relational `<`, `<=`, `>` or `>=` expressions add to the set of overload candidates _rewritten_ expressions involving the same relational operator on the result of a call to a three-way operator: `a < b` is rewritten as `(a <=> b) < 0`, `a >= b` is rewritten as `(a <=> b) >= 0`, and so on. Name lookup is performed for `operator<=>` as it would be for the four relational operators.
* Evaluation of the three-way expression `<=>` adds to the set of overload candidates _synthesized_ expressions, in which the order of arguments is reversed: `a <=> b` adds a synthesized candidate `0 <=> (b <=> a)`. Swapping the arguments of `<=>` reverses the result of the comparison (so that _less_ becomes _greater_) and running that result through `0 <=> result` reverses them back. Synthesized expressions are also considered indirectly during overload resolution for _rewritten_ relational operators: `a < b` is rewritten as `(a <=> b) < 0`, and that in turn considers the _synthesized_ expression `0 < (b <=> a)`.

## Rewritten equivalence operators

* Evaluation of the `!=` expression adds to the set of overload candidates _rewritten_ expressions involving `operator==`. That is, `a != b` is rewritten as `!(a == b)`, and these candidates are also considered during overload resolution. 
* Evaluation of the `==` expression adds to the set of overload candidates _synthesized_ expressions, in which the order of arguments is reversed: `a == b` adds a synthesized candidate `b == a`. Synthesized expressions are also considered indirectly during overload resolution for _rewritten_ `!=` operators: `a == b` is rewritten as `!(a == b)`, and that in turn considers the _synthesized_ expression `!(b == a)`. 

This is one of the few times C++ has added functionality on something other than an opt-in basis, as the rewritten and synthesized candidates are considered even in the absence of the user taking any action. For the relational operators, the user at least has to declare an `operator<=>` for name lookup to find a _rewritten_ candidate. For overload resolution on `!=`, existing `operator==` functions become candidates, possibly making an ill-formed C++17 program well-formed under C++20.

## Overload resolution

For both relational and equivalence operators, rewritten candidates are worse (for purposes of overload resolution) than non-rewritten candidates. Synthesized candidates are worse than non-synthesized candidates. However, the tests for rewritten and synthesized candidate tests come late in the overload resolution ranking process. Before these properties are considered, functions will be compared:
1. according to to the rank of implicit conversion sequences for converting each argument to its parameter type,
1. non-template functions are preferred over function templates,
1. and more-specialized function templates are preferred over less-specialized function templates (according to partial ordering rules).

If two functions are equivalent even after this point, then the rewritten/synthesized tests are applied.


