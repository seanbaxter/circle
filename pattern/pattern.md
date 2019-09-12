# Pattern-matching expressions and enhanced structured bindings

_Pattern matching_ is a domain-specific language embedded in a general-purpose language, intended to both test an expression and to extract information from the expression. A _regular-expression parser_ is a pattern-matching tool intended to process text:
* An input is tested against conditions, such as the pattern signifying a street address.
* Components of the input are extracted and bound to variables, such as the street number, street name, postal code and so on.

Pattern matching in a general-purpose language accomplishes much the same thing: it matches an expression against a test, and extracts data by binding variables to expression components. A pattern match statement or expression addresses the organizational challenges of long _if-else_ chains (allowing more expressiveness in the clause than the _if-statement_'s _condition_ rule), while providing much greater flexbility compared to the usual _switch-statement_. It's not unhelpful to consider the pattern match as a "switch on steroids."

[**pattern1.cxx**](pattern1.cxx)
```cpp
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if(2 != argc) {
    printf("Give me a number\n");
    return -1;
  }
  
  long x = atol(argv[1]);
  @match(x) {
    1                       => printf("It's 1\n");
    < 0                     => printf("It's negative\n");
    > 100                   => printf("More than 100\n");
    2 ... 5                 => printf("A number 2 <= x < 5\n");
    5 ... 10 && !7          => printf("5 <= x < 10 but not 7\n");
    7 || 10 || 13           => printf("7 or 10 or 13\n");
    10 ... 15 || 20 ... 25  => printf("In disjoint ranges\n");
    ! 30 ... 90             => printf("Not between 30 and 90\n");
    _x if(1 & _x)           => printf("%d is an odd number\n", _x);
    < 50 _x if(0 == _x % 4) => printf("Less than 50 but multiple of 4\n");
    _                       => printf("Everything else\n");
  };
  
  return 0;
} 
```
```
$ circle pattern1.cxx

$ ./pattern1 9
5 <= x < 10 but not 7

$ ./pattern1 36
Less than 50 but multiple of 4

$ ./pattern1 99
Not between 30 and 90

$ ./pattern1 39
39 is an odd number
```

The pattern match is compact like a switch, but has all the expressiveness of an _if-statement_. Each semicolon-delimeted statement in the match's brackets comprise a _clause_. The clause has a pattern on the left followed by an optional _guard-expression_ (introduced by the `if` token), a fat arrow **=>** in the middle, and a statement or expression on the right. The clauses are tried from top-to-bottom. If the pattern matches, then the corresponding statement is executed (for _match-statement_) or corresponding expression is returned (for _match-expression_). 

A pattern is specified with a different syntax from the rest of C++, allowing us to mix declarations and expressions in a fine-grained way. In the pattern `< 50 _x if(0 == _x % 4)`, the initializer for the pattern (which is an lvalue to an unnamed object initialized with the result object of `atoi(argv[1]))`) is first tested against 50. Having passed the test, an object `_x` is declared in the scope of the clause and bound to that initializer. We're now at the guard expression, which tests `0 == _x % 4`. If this returns true, then we execute the attached statement and break out of the _match-statement_.

[**pattern2.cxx**](pattern2.cxx)
```cpp
#include <iostream>

struct Player { std::string name; int hitpoints; int coins; };

void get_hint(const Player& p) {
  @match(p) {
    [.hitpoints: 1] => std::cout << "You're almost destroyed. Give up!\n";
    [.hitpoints: 10, .coins: 10] => std::cout << "I need the hints from you!\n";
    [.coins: 10] => std::cout << "Get more hitpoints!\n";
    [.hitpoints: 10] => std::cout << "Get more ammo!\n";
    [.name: _n] => {
      if (_n != "The Bruce Dickenson") {
        std::cout << "Get more hitpoints and ammo!\n";
      } else {
        std::cout << "More cowbell!\n";
      }
    }
  };
}

int main() {
  get_hint(Player { "Batman", 10, 15 });
  get_hint(Player { "Spider-man", 5, 10 });
  get_hint(Player { "Aquaman", 10, 10 });
  get_hint(Player { "Iron Man", 5, 3 });
   
  return 0;
}
```
```
$ circle pattern2.cxx
$ ./pattern2
Get more ammo!
Get more hitpoints!
I need the hints from you!
Get more hitpoints and ammo!
```

By combining matches with structured and designated bindings, we're able to extract a lot of information from an input without a lot of typing, similar to a regular expression. This example was lifted from [p1371r1](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1371r1.pdf), and demonstrates designated bindings in a _match-statement_.

## References

A detailed overview of pattern matching:
[Pattern Matching: Match Me If You Can by Michael Park](https://www.youtube.com/watch?v=nOwUzFYt0NQ)

Pattern Matching C++ proposal by Michael Park, which got me started on my own extension:
[p1371r1](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1371r1.pdf)

How Rust does it:
[Overview of Rust's pattern matching](https://doc.rust-lang.org/book/ch18-03-pattern-syntax.html)

## Designated bindings and enhanced structured bindings

C++17 introduced _structured bindings_, a new declaration syntax for decomposing arrays and class objects into their constituent members. Circle greatly extends this feature, allowing recursive structured bindings as well as designated bindings, which are declarations that bind to member names rather than ordinals.

### C++17 structured bindings

[**binding1.cxx**](binding1.cxx)
```cpp
#include <tuple>
#include <map>
#include <string>

void cxx17_structured_bindings() {

  // Structured binding (positional) to public non-static data members x, y, z.
  struct foo_t {
    int x, y, z;
  };  
  foo_t obj1 { 5, 6, 7 };
  auto& [a1, b1, c1] = obj1;
  printf("%d %d %d\n", a1, b1, c1);
  
  // Structured binding on tuple-like object.
  std::tuple<int, double, const char*> obj2 {
    10,
    3.14,
    "a very long string"
  };
  auto& [a2, b2, c2] = obj2;
  printf("%d %f %s\n", a2, b2, c2);

  // Structured binding on array.
  int array3[] { 10, 20, 30, 40 };  
  auto [a3, b3, c3, d3] = array3;
  printf("%d %d %d %d\n", a3, b3, c3, d3);

  // Structured binding in a ranged-for loop. Each element of map is an 
  // std::pair, which is "tuple-like" by C++'s definitinon. The structured
  // binding uses std::get<0> and std::get<1> to decompose the pair into the
  // [key, value] declarations.
  std::map<int, std::string> map {
    { 1, "One" }, { 2, "Two" }, { 3, "Three" }
  };
  for(auto& [key, value] : map)
    printf("%d : %s\n", key, value.c_str());
}

int main() {
  cxx17_structured_bindings();
  return 0;
}
```

C++17 supports structured-binding declarations that are specified with an _identifier-list_, as above. An implicitly-declared structured-binding objects is created and initialized with the right-hand side of the structured-binding declaration. If a _ref-qual_ is present, this implicit object is an lvalue or rvalue reference; otherwise it's an object reference. 

If the object is an array, each identifier in the _identifier-list_ is bound to one of the array elements. If the object is a tuple-like object (meaning `std::tuple_size<type>` yields a complete object, where `type` is the type of the initializer expression), each binding is initialized with the result object of `std::get<I>(object)`, where I is the ordinal of the binding. Otherwise, the initializer must be a class object, and each identifier binds to a public non-static data member.

### Enhanced structured bindings

Circle enhances the structured binding by throwing out the _identifier-list_ syntax and adopting a pattern syntax. This freer syntax gives three new features:
1. **Recursive binding declarations:** Each element in the structured binding can be a new pattern, rather than merely an identifier. This allows continuous decomposition of elements that are themselves aggregates.
1. **Designated bindings:** Specify bindings by member name rather than position. This resembles member access, which is more idiomatic to C programmers than indexed access.
1. **Wildcard bindings:** Use the underscore `_` token to specify a wildcard token. It excuses the user from having to come up with new names for each binding, and is intended to pad out elements in a structured binding that the user doesn't need. This declaration has no associated type or value.

[**binding2.cxx**](binding2.cxx)
```cpp
#include <cstdio>

void circle_enhanced_bindings() {
  // Declare a designated binding. This binds according to member name
  // instead of position within an aggregate. The names do not have to be
  // ordered according to the data member declarations.
  struct vec4_t {
    int x, y, z, w;
  };
  vec4_t obj { 100, 200, 300, 400 };

  // Bind only the .x and .z components using designated bindings.
  auto& [.x : x1, .z : z1] = obj;
  printf("x1 = %d, z1 = %d\n", x1, z1);

  // Bind only the .x and .z components using wildcards.
  auto& [x2, _, z2, _] = obj;
  printf("x2 = %d, z2 = %d\n", x2, z2);

  // Declare a recursive structured-binding pattern to decompose a 2D
  // array. This is not allowed by C++17, because it only accepts 
  // identifier-list bindings.
  int array[][3] {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  auto& [ [m11, m12, m13], [m21, m22, m23], [m31, m32, m33] ] = array;
  printf("matrix = <%d, %d, %d>, <%d, %d, %d>, <%d, %d, %d>\n", 
    m11, m12, m13, m21, m22, m23, m31, m32, m33);

  // Use both structured and designated bindings to extract the .w 
  // members from each vector.
  vec4_t vecs[] {
    { 10, 11, 12, 13 },
    { 20, 21, 22, 23 },
    { 30, 31, 32, 33 }
  };
  auto& [ [.w : w1], [.w : w2], [.w : w3] ] = vecs;
  printf("w1 = %d, w2 = %d, w3 = %d\n", w1, w2, w3);
}

int main() {
  circle_enhanced_bindings();
  return 0;
}
```
```
$ circle binding2.cxx 
$ ./binding2
x1 = 100, z1 = 300
x2 = 100, z2 = 300
matrix = <1, 2, 3>, <4, 5, 6>, <7, 8, 9>
w1 = 13, w2 = 23, w3 = 33
```

Keep in mind that `[[` is its own token in C++, which signifies the start of an attribute. Be careful to keep a space between the `[` tokens when introducing nested structured bindings:

* `auto& [[m11, m12], [m21, m22]] = array;` This is a parse error due to the token `[[`.
* `auto& [ [m11, m12], [m21, m22] ] = array;` This parses as expected.

## Enhanced bindings in pattern matching

Structured and designated patterns, when used in a match expression or statement, split the initializer expression into components, which in turn may be tested against expressions and bound to clause-scoped declarations. But in the context of pattern matching, binding presents us an ambiguity:

[**pattern3.cxx**](pattern3.cxx)
```cpp
#include <cstdio>

int main() {

  struct foo_t {
    int x, y, z;
  };
  foo_t obj { 3, 4, 5 };

  int Z = 6;
  @match(obj) {
    // Test an expression against the initializer.
    [_, _, 3]    => printf(".z is 3\n");  // structured binding
    [  .z: 4]    => printf(".z is 4\n");  // designated binding

    // Is Z a test/expression or a binding? If the clause fails, it's got to
    // be a test.
    [_, _, Z]    => printf("Z must be a binding\n");
    _            => printf("Z must be an expression\n");
  };

  return 0;
}
```
```
$ circle pattern3.cxx
$ ./pattern3
Z must be an expression
```

In the first two clauses, 3 and 4 are clearly intended as expressions to test the .z member of the input. But what if we stored the test value in an object and named it inside the pattern. Is `Z` a test, or is `Z` a binding declaration? [Park et al](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1371r1.pdf) take the latter view, and treat anything that can be a binding as a binding. To interpret `Z` as a test in their world, place it after a `case` token. That is, `case Z` is a test, and `Z` is a binding.

I find the use of disambiguating tokens rather troublesome; C++ already uses `typename` and `template` as disambiguating tokens when dealing with dependent types and dependent member expressions, respectively, and even very experienced C++ programmers commonly flub these usages. For this first cut of pattern matching in Circle, identifiers _with leading underscores_ signify bindings; other identifiers signify expressions. This also reduces visual noise in patterns.

For example, 
* `_` is a wildcard.
* `_x` is a binding.
* `x` is an expression.
* `_x + _y` is an expression.
* `this->_x` is an expression. `this->` disambiguates.
* `(_x)` is an expression. `()` disambiguates.

I think this is the natural choice for most users. The underscore reinforces that one is dealing with a binding, while eliminating the need for disambiguation in most cases.

As with [p1371r1](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1371r1.pdf), the binding declarations in patterns bind references to the initializers, not value types. They're similar to using `auto& [x, y, z] = init;` in a structured binding. If you write to a binding declaration in a pattern, you write to its underlying object member.

[**pattern4.cxx**](pattern4.cxx)
```cpp
#include <cstdio>

int sq(int x) {
  return x * x;
}

int main() {
  struct foo_t {
    int x, y, z;
  };
  foo_t obj { 3, 4, 7 };

  @match(obj) {
    [_x, _y, sq(_x) + sq(_y)]                   => printf("Sum of squares!\n");
    [_x, _y, abs(sq(_x) - sq(_y))]              => printf("Difference of squares!\n");
    [_x, _y, _z] if(sq(_x) + sq(_y) == sq(_z))  => printf("Perfect squares!\n");
    _                                           => printf("I got nothing.\n");
  };
  return 0;
} 
```
```
$ circle pattern4.cxx
$ ./pattern4
Difference of squares!
```

## Pattern matching test syntax

There are six kinds of patterns:
1. Wildcards: the underscore identifier.
1. Bindings: other underscore-leading identifiers.
1. Structured bindings.
1. Designated bindings.
1. Tests.
1. Dereference operator.

Wildcards and bindings are terminals in this grammar. Structured and designated bindings are non-terminals, as their elements are recursively parsed as patterns. Tests can go either way: a test by itself is a terminal, but a test may precede another pattern.

Tests have their own grammar. The operators from lowest-to-highest predence are:

* `||` has lowest precedence.
* `&&`
* `!`
* Test expressions are the terminals of the test grammar.

### Comparison tests

Binary expressions cover the four C++ comparison operators, `<`, `<=`, `>` and `>=`. The pattern initializer is implicitly placed on the left-hand side, and an _inclusive-or-expression_ (that is, expressions involving operators with the same precedence as bitwise-OR `|` or higher) is specified by the user on the right-hand side.

* `< 0` - compare the pattern's initializer to 0.
* `< 3 + 7` - compare to the expression `3 + 7`.
* `< 3 || 7` - the initializer is less than 3, or it's 7. 
* `< 10 && !5` - the initializer is less than 10 but not 5.

In C++, it's idiomatic to provide only `operator<` for user-defined types, so each of the four comparisons are actually transformed to calls to `<` and `!`:

* `a < b` is itself
* `a <= b` is transformed to `!(b < a)`
* `a > b` is transformed to `b < a`
* `a >= b` is transformed to `!(a < b)`

The negation operator `!` is not allowed before comparison tests, as it could only confuse the user. Choose the operator with the comparison you want to effect, and it will be transformed by the compiler to a call to `<`.

### Conditional test

The conditional test `?` performs contextual conversion to bool on the pattern initializer. If the result is true, the test passes. This may be used in conjunction with `!` to test that the initializer is null or false.

* `[.y: ?]` - test that the `y` data member converts to true.
* `[_, _, !?]` - test that the third aggregate element converts to false.

### Equivalence test

If the test doesn't begin with a comparison token or `?`, it's interpreted as an expression. This is an _inclusive-or-expression_ in the place of the pattern. The pattern initializer is implicitly compared to this expression using (the perhaps overloaded) operator `==`. The value of the expression itself doesn't matter, only how it compares to the pattern initializer.

* `3` - the initializer is 3.
* `3 || > 10 && < 50` - the initializer is 3, or it's greater than 10 and less than 50.

Due to the expression test covering only operators with the precedence of `|` and higher, the `||` and `&&` operators in this second pattern are processed by the pattern parser, not the compiler's usual expression parser. This treatment allows stringing multiple tests together in one pattern. If you feel the need to use parenthesis to change precedence, it's probably best to just create a binding and use the guard expression that comes after the pattern in the match clause.

If you want your test to override the pattern's treatment of these operators, enclose your test expression in `()`. But note the top-level test remains, and the expression is compared against the pattern's initializer expression, and not just tested against true/false. 

* `3 || > 10 && < 50` - the initializer is 3, or it's greater than 10 and less than 50.
* `(3 || > 10 && < 50)` - a syntax error, because the tokens inside `()` are parsed as an _expression_.
* `3 || !4 && !10` - the initializer is 3, or it something other than 4 or 10.
* `(3 || !4 && !10)` - `3 || !4 && !10` evaluates to true. compare the initializer to true.

### Range test

If the token immediately after an equivalence test is `...`, a second expression test is immediately parsed and a range test is formed. The grammar is _inclusive-or-expression_ `...` _inclusive-or-expression_. If `x` refers to the pattern's initializer, then the range `a ... b` conceptually evaluates `a <= x && x < b`. However, it's implemented as `!(x < a) && (x < b)` to support user-defined types with an overloaded `operator<`.

* `1 ... 10` - 1 <= x < 10.
* `!1 ... 10` - not in the range 1 <= x < 10.
* `1...10` - a tokenization error. 1. looks like the start of a floating-point number, but isn't valid.
* `0 ... 5 || 10 ... 15` - in the range 0 <= x < 5 or 10 <= x < 15.

### Expression test

All the tests above compare the pattern initializer expression to something. What if, instead, the initializer should be an argument to an expression, and that expression is itself the test? We introduce the _expression test_ after the `/` token. But now that the initializer isn't implicitly on the left-hand side of a comparison, but is rather part of a _condition_ expression, we need a way to access the initializer's value prior to binding it.

In the context of pattern tests, the underscore `_` is a special declaration that holds the pattern initializer.

All the above forms can be rewritten using expression tests:

* `5` is the same as `/ _ == 5`
* `< 10` is the same as `/ _ < 10`
* `1 ... 5` is the same as `/ (!(_ < 1) && (_ < 5))`
* `?` is the same as `/ (bool)_`

[**pattern4_1.cxx**](pattern4_1.cxx)
```cpp
#include <cstdio>

int sq(int x) {
  return x * x;
}

int main() {
  struct foo_t {
    int x, y, z;
  };
  foo_t obj { 3, 4, 5 };

  // Use / to evaluate an expression test. The _ token inside any pattern test
  // gives the pattern initializer at that point.
  @match(obj) {
    // Compare .z to expressions of _x and _y.
    [_x, _y, sq(_x) + sq(_y)]                   => printf("Sum of squares!\n");
    [_x, _y, abs(sq(_x) - sq(_y))]              => printf("Difference of squares!\n");
    
    // We can bind _z to .z and use a guard
    // [_x, _y, _z] if(sq(_x) + sq(_y) == sq(_z))  => printf("Perfect squares!\n");

    // or we can use / to introduce an expression test. The _ declaration in
    // a pattern test refers to the initializer for that element, in this case
    // .z. We can optionally bind the .z member after the pattern test.
    [_x, _y, / sq(_x) + sq(_y) == sq(_)]        => printf("Perfect squares!\n");

    _                                           => printf("I got nothing.\n");
  };
  return 0;
} 

```
```
$ circle pattern4_1.cxx
$ ./pattern4_1
Perfect squares!
```

Here we amend [pattern4_1.cxx](pattern4_1.cxx). The perfect squares test previously required a guard expression, because we needed the value of the `z` member to square and compare it. Using the expression test allows us to compare an expression not to the initializer, but simply to test its true/false status. Exposing `_` lets us incorporate the yet-unbound pattern initializer into this conditional expression.

## Dereference operator

The `*` token at the start of a pattern applies the indirection operator to the pattern's initializer. This is useful for dereferencing a pointer scalar or smart-pointer user-defined type to access the lvalue of the pointed-at object. This operator is not a test, but it should be used in conjunction with the `?` test, which checks that a pointer is not null prior to dereferencing and binding its pointed-at members.

[**pattern5.cxx**](pattern5.cxx)
```cpp
#include <cstdio>

int main() {
  struct node_t {
    int x, y;
    node_t* p;
  };

  node_t a { 1, 2, nullptr };
  node_t b { 3, 4, &a };

  @match(b) {
    [.p: ? * [_x, _y]] => printf("p->x = %d, p->y = %d\n", _x, _y);
    [_x, _y, _] => printf("x = %d, y = %d, p = null\n", _x, _y);
  };

  return 0;
}
```
```
$ circle pattern5.cxx
$ ./pattern5
p->x = 1, p->y = 2
```

In the first clause, the initializer is a node_t lvalue (refering to `b`). The designated binding `.p` accesses `b.p`, yielding a pointer lvalue. This initializer is passed through the `?` test, which performs contextual conversion to bool on the pointer. Because the pointer isn't nullptr, this check succeeds, and matching within the clause continues. The `*` operator is encountered, which applies `operator*` on the initializer, yielding `*b.p`. Now we're at a structured binding which gets initialized not with the value of `b`, but with `a`! We bind `a.x` and `a.y` and print them out. If the `b.p` pointer were null, it would fail the test, and the second clause would match, causing execution of the second statement.

## Match expressions and match statements

`@match` constructs comes in two forms: expressions and statements. Match expressions may be used as subexpressions, and each clause returns a result object for the overall pattern. The right-hand side of each clause must specify an expression statement, and not any other kind of statement. Statement expressions may have any kind of statement to the right of the **=>**.

`@match` constructs are treated as statements, unless there are preceding or trailing tokens, indicating that it's part of a larger expression. Match expressions take an optional _trailing-return-type_ to implicitly convert result expressions to the match-expression's return type. This is similar to the use of _trailing-return-type_ in a lambda function, which may have multiple return statements.

[**pattern6.cxx**](pattern6.cxx)
```cpp
#include <cstdio>

int main() {
  struct foo_t {
    int x;
    long y;
    float z;
  };
  foo_t obj { 4, 5, 6 };

  double x = 2 + @match(obj) -> float {
    // Implicitly cast each return expression to float.
    [_x, _, 5] => _x;    // If z == 5, return x.
    [_, 5, _z] => _z;    // If y == 5, return z.
    [5, _y, _] => _y;    // If x == 5, return y.
    _          => 0;     // Else, return 0.
  } / 3;
  
  printf("%f\n", x);
  return 0;
}
```
```
$ circle pattern6.cxx
$ ./pattern6
4.000000
```

## Meta control flow in pattern matching

[Park et al](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1371r1.pdf) propose using commas to separate clauses in match expressions and semicolons to separate clauses in match statements. They also propose using colon to separate the left-hand side pattern from the right-hand side statement in a match statement, and a fat arrow **=>** to separate the left-hand side pattern from the right-hand side expression in a match expression. 

These choices create parser difficulties for Circle's treatment of pattern matching. Unlike the C++ proposal, Circle's clauses are _parsed as ordinary statements_. That is, inside the match braces, the tokens are parsed as they would be inside the braces of a _compound-statement_ or _enum-_ or _class-specifier_. While the only valid _real_ statement is a match clause (i.e., a pattern on the left, fat arrow, statement or expression on the right), meta statements are still permitted as a mechanism for code generation. 

[**pattern7.cxx**](pattern7.cxx)
```cpp
#include <cstdio>
#include <string>
#include <stdexcept>

template<typename type_t>
std::string enum_to_string(type_t e) {
  switch(e) {
    // A compile-time loop inside a switch.
    @meta for enum(type_t e2 : type_t) {
      case e2: 
        return @enum_name(e2);
    } 
  }
}

int main() {

  enum class shape_t {
    circle,
    square, 
    triangle,
    hexagon,
  };

  struct foo_t {
    double radius;
    shape_t shape;
  };

  foo_t obj { 3.0, shape_t::triangle };

  const char* s = @match(obj) {
    // Compile-time loop over each enum in shape_t.
    @meta for enum(auto shape : shape_t) {

      // Compile-time loop over the pairs in this array. Use structured
      // bindings for r_limit and size.
      @meta std::pair<double, std::string> sizes[] {
        { 1.0, "small" }, { 5.0, "medium" }, { -1, "large" }
      };
      
      @meta for(auto& [r_limit, size] : sizes) {
        // Form an std::string concatenating our message as a compile-time 
        // object. Use @string to convert it to a string literal available
        // at runtime.
        @meta std::string s = "A " + size + " " + enum_to_string(shape);
        @meta printf("%s\n", s.c_str());

        // If -1 != r_limit, test against the radius and the shape. Otherwise
        // test only against the shape.
        @meta if(-1 != r_limit)
          // Match only when radius < r_limit.
          [.radius: < r_limit, .shape: shape] => @string(s);
        else
          [.shape: shape] => @string(s);
      }
    }

    // Create a default.
    _ => "Unrecognized shape";
  };

  printf("%s\n", s);
  return 0;
}
```
```
$ circle pattern7.cxx
A small circle
A medium circle
A large circle
A small square
A medium square
A large square
A small triangle
A medium triangle
A large triangle
A small hexagon
A medium hexagon
A large hexagon

$ ./pattern7
A medium triangle
```

We can use any meta statements inside the _match-expression_, as we do in the _enum-specifier_ that begins this sample. First we loop over the enumerators in `shape_t`. Inside that we define an array defining small, medium and large radius limits, and meta for over that. The indices are concatenated into a `std::string` _at compile time_. We branch over `r_limit` to conditionally include a comparison test in the pattern if `r_limit` isn't -1. The right-hand side of the clause returns `@string(s)`, which is the string literal version of the compile-time concatenated string. Normally we'd need to include a _trailing-return-type_ in the _match-expression_ to allow returning different types, but string literals are array types, and when returned from functions arrays decay to pointer types, avoiding return-type-deduction failure.

## Patterns involving types

[p1371r1](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1371r1.pdf) proposes some "alternative patterns," which name types inside `< >`. They name several cases for this pattern:

1. `std::variant`-like. If `std::variant_size_v<V>` is well-formed, convert the initializer to the requested type using `std::variant_alternative_t<I, V>`.
1. `std::any`-like. If there exists a valid non-member `any_cast<Alt>`, use that to convert to the requested type.
1. Polymorphic types. If `std::is_polymorphic_v<V>` is true, use `dynamic_cast` to convert to the requested type.

I'm holding off on implementing type-oriented patterns like this on my first cut at pattern matching. The reason is that Circle has a special mechanism for dealing with types, the [typed enum](https://github.com/seanbaxter/circle/blob/master/examples/README.md#typed-enums), which makes implementing variant types very easy. Adding support matching types against typed enums would be very easy, and also allow convenient meta for generation of clauses. 

However, to make the alternative patterns really generic, some thought needs to go into their treatment under substitution failure. When the clause is instantiated and a type specified in an alternative pattern is not supported by the initializer, does this make the program ill-formed, or is the clause silently dropped from the match definition? I feel that SFINAE behavior during match instantiation makes the construct more expressive, although this instantiation behavior is without precedent--a _case-statement_ which fails to substitute, or a condition in an _if-else_ chain, always generates a compilation error. Coming from the other direction, since Circle lets you define a match definition with compile-time control flow, couldn't that be used to selectively add clauses appropriate to each type _at instantiation_? It would be a mistake on the part of the user to generate an alternative pattern incompatible with the _match-expression_'s initializer type.
