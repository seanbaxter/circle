# Circle conditionals

Circle supplements Standard C++ by adding three more conditional operators:

* constexpr conditional `?? :`
* multi conditional `... ? :`
* constexpr multi conditional `... ?? :`

## Constexpr conditional `?? :`

Constexpr conditionals are the expression equivalent of `if-constexpr/else` constructs. During substitution of the expression `a ?? b : c`, `a` is evaluated in a constexpr context; it's value must be resolved at compile time. Then, either `b` or `c` is substituted. The branch not taken is dismissed. None of the type conversion semantics of [[expr.cond]](http://eel.is/c++draft/expr.cond) are applied here; the `b` or `c` branch is substituted and returned without adjustment.

[**call1.cxx**](call1.cxx)
```cpp
#include <iostream>

void call(auto f, const auto& x) {
  requires { f(x); } ?? 
    std::cout<< f(x)<< "\n" : 
    std::cout<< "Could not call f("<< @type_string(decltype(x))<< ")\n";
}

int f(int x) { return x * x; }

int main() {
  call(f, 5);
  call(f, "Hello");
}
```
```
$ circle call1.cxx && ./call1
25
Could not call f(const char[6])
```

_requires-expression_ is a C++20 concept feature. During substitution it yields true if the contained expressions can be formed, and false otherwise. This sample uses it to test if `f` is callable with `x`. If it is, the constexpr conditional emits the `f(x)` call. If it isn't, it produces an error message.

As with _if-constexpr_ statements, this operator only protects you from ill-formed expressions in a dependent context. Outside of a template definition, writing `f(x)` when that function isn't callable will raise a compile-time error.

## Multi conditional `... ? :`

The multi conditional operator defines a new expansion locus for programmatically chaining together multiple conditional clauses. In the expression `a ... ? b : c`, `a` must and `c` must not contain an unexpanded parameter pack. Expansion of `a` recursively defines conditionals, until the parameter is pack is exhausted, at which point `c` is yielded. If `b` contains an unexpanded parameter pack (very likely), then it is expanded in conjunction with `a`.

[**call2.cxx**](call2.cxx)
```cpp
#include <tuple>
#include <iostream>

template<typename func_t, typename... types_t>
auto call_tuple1(func_t f, const std::tuple<types_t...>& tuple, int index) {
  switch(index) {
    @meta for(int i : sizeof...(types_t)) {
      case i:
        return f(tuple...[i]);
    }
  }
}

template<typename func_t, typename... types_t>
void call_tuple2(func_t f, const std::tuple<types_t...>& tuple, int index) {
  // Like the above, but one line.
  return int... == index ...? f(tuple...[:]) : __builtin_unreachable();
}

int main() {
  auto f = [](const auto& x) {
    std::cout<< x<< "\n";
  };

  auto tuple = std::make_tuple(1, 5.5, "Hello tuple");

  call_tuple1(f, tuple, 0);
  call_tuple2(f, tuple, 1);
  call_tuple2(f, tuple, 2);
}
```
```
$ circle call2.cxx && ./call2
1
5.5
Hello tuple
```

This powerful visitor pattern calls the function-like object `f` with the tuple element held at `index`. We can't dynamically access elements of tuples (or variants), because their heterogeneity requires static address calculations. Before multi conditional, programmatically generating a switch would be the course.

With the availability of multi conditional, we can generate equivalent code inside a single expression. First, notice the [tuple slice operator `...[:]`](https://github.com/seanbaxter/circle/blob/master/universal/README.md#static-subscripts-and-slices). This internally calls `std::get<I>(tuple)` for each element of the tuple, yielding a non-type parameter pack. This pack expression is passed to the candidate function `f`.

Because the `b` expression in `a ...? b : c` is a parameter pack, we can use its size to infer the size of the integer pack `int...` in the `a` condition. `int... == index` is a pack of expressions, `0 == index`, `1 == index` and so on. This generates a test of the incoming index against each tuple element index. When there's a match, the corresponding function call subexpression is evaluated.

Assume the index is always valid. ALthough one of function call subexpressions will always be taken, but we're still obligated to terminate the multi condition operator with a `c` expression. In this case, we call into `__builtin_unreachable()`. This informs the compiler of undefined behavior. It makes the code equivalent to a switch statement with all valid cases covered, but without a default target. The semantics of [[expr.cond]](http://eel.is/c++draft/expr.cond) were modified to accommodate `[[noreturn]]` expressions like `__builtin_unreachable` in addition to _throw-expressions_. This was independently proposed by [Ternary Right Fold Expression P1012R1](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p1012r1.pdf) by Frank Zingsheim.

[[*call3.cxx*]](call3.cxx)
```cpp
template<int x> int func();

int call(int index) {
  return int...(4) == index ...? func<int...>() : __builtin_unreachable();
}
```
```ll
define i32 @_Z4calli(i32) local_unnamed_addr {
  switch i32 %0, label %4 [
    i32 0, label %10
    i32 1, label %8
    i32 2, label %6
  ]

; <label>:2:                                      ; preds = %6, %4, %8, %10
  %3 = phi i32 [ %11, %10 ], [ %9, %8 ], [ %7, %6 ], [ %5, %4 ]
  ret i32 %3

; <label>:4:                                      ; preds = %1
  %5 = tail call i32 @_Z4funcILi3EEiv()
  br label %2

; <label>:6:                                      ; preds = %1
  %7 = tail call i32 @_Z4funcILi2EEiv()
  br label %2

; <label>:8:                                      ; preds = %1
  %9 = tail call i32 @_Z4funcILi1EEiv()
  br label %2

; <label>:10:                                     ; preds = %1
  %11 = tail call i32 @_Z4funcILi0EEiv()
  br label %2
}
```

We can be encouraged that the LLVM backend exploits `__builtin_unreachable` to produce better code. In this test, the branch to the `__builtin_unreachable` is eliminated (after all, it is unreachable) and the switch's default target jumps to the last conditional element. At least for simple examples, we're not leaving any performance on the table by using the multi conditional operator over the more verbose switch-building constructs.

```cpp
#include <cstdio>
#include <iostream>

template<typename enum_t>
const char* enum_to_name1(enum_t e) {
  switch(e) {
    @meta for enum(enum_t e2 : enum_t) {
      case e2:
        return @enum_name(e2);
    }
    default:
      return "<unknown>";
  }
}

template<typename enum_t>
const char* enum_to_name2(enum_t e) {
  return @enum_values(enum_t) == e ...? @enum_names(enum_t) : "<unknown>";
}

enum shapes_t {
  circle, square, rectangle = 100, octagon
};

int main() {
  std::cout<< enum_to_name1(square)<< "\n";
  std::cout<< enum_to_name2(rectangle)<< "\n";
  std::cout<< enum_to_name2((shapes_t)102)<< "\n";
}
```
```
$ circle enum.cxx && ./enum
square
rectangle
<unknown>
```

The multi conditional operator is aided by Circle's exhaustive suite of [pack-yielding intrinsics](https://github.com/seanbaxter/circle/blob/master/reflection/README.md#introspection-reference). The `enum_to_name` routine is shrunk down to a one-liner by coordinating the `@enum_values` and `@enum_names` intrinsics. 

## Constexpr multi conditional `... ?? :`

The constexpr multi conditional combines both of the earlier primitives. Like the constexpr expression, the condition subexpression is evaluated at compile time and only the corresponding branch is emitted; there is no conversion to a common type. Like the multi conditional, expanding a pack condition expression chains the operation.

[**call_first.cxx**](call_first.cxx)
```cpp
#include <iostream>
#include <array>
#include <utility>

auto call_first(auto&& x, auto&&... fs) {
  return requires { fs(x); } ...?? 
    fs(x) :     
    static_assert(@type_string(decltype(x)));
}

void f1(double x) { std::cout<< "f1: "<< x<< "\n"; }

void f2(const char* x) { std::cout<< "f2: "<< x<< "\n"; }

auto f3 = []<typename type_t, size_t I>(std::array<type_t, I> a) {
  std::cout<< "f3: ";
  std::cout<< a...[:]<< " " ...;
  std::cout<< "\n";
};

int main() {
  call_first("Hello ??", f1, f2, f3);

  // This causes an error:
  // call_first(std::pair(1, 2), f1, f2, f3);
}
```
```
$ circle call_first.cxx && ./call_first
f2: Hello ??
```

The `call_first` utility takes a value `x` and a pack of functions/function objects `fs`. It attempts to call them in sequence from inside a _requires-expression_, until it finds one that works, then emits the call and returns its result object. If the _requires-expression_ is false for all candidates, the new _static_assert_ expression is invoked, which helps document the nature of the failure. This kind of expression always breaks the build when compiled in a non-dependent context. Because of the if-constexpr nature of `??:` and `...??:`, we can guard against its substitution. 

[**visit.cxx**](visit.cxx)
```cpp
#include <variant>
#include <tuple>
#include <array>
#include <iostream>

template<typename type_t>
auto call_first(const type_t& x, auto&&... fs) {
  return requires { fs(x); } ...?? fs(x) : static_assert(@type_string(type_t));
}

template<typename... types_t, typename... funcs_t>
auto visit1(const std::variant<types_t...>& variant, funcs_t&&... fs) {
  return int...(sizeof...(types_t)) == variant.index() ...?
    call_first(std::get<int...>(variant), std::forward<funcs_t>(fs)...) :
    __builtin_unreachable();
}

template<typename... types_t>
auto visit2(const std::variant<types_t...>& variant, auto&&... fs) {
  switch(variant.index()) {
    @meta for(int i : sizeof...(types_t)) {
      case i:
        return requires { fs(std::get<i>(variant)); } ...?? 
          fs(std::get<i>(variant)) :
          static_assert(@type_string(types_t...[i]));
    }
  }
}


void f1(double x) { std::cout<< "f1: "<< x<< "\n"; }

void f2(const char* x) { std::cout<< "f2: "<< x<< "\n"; }

auto f3 = []<typename type_t, size_t I>(std::array<type_t, I> a) {
  std::cout<< "f3: ";
  std::cout<< a...[:]<< " " ...;
  std::cout<< "\n";
};

int main() {
  std::variant<
    double,
    const char*,
    std::array<int, 3>,
    std::array<double, 2>
  > v;
  
  v = 3.14;
  visit1(v, f1, f2, f3);

  v = "Hello";
  visit1(v, f1, f2, f3);
  
  v = std::array { 1, 2, 3 };
  visit2(v, f1, f2, f3);
}
```

Here we take the `call_first` routine and turn it into a full-blown variant visitor pattern. Instead of passing an item to `call_first` we pass an `std::variant` to `visit1` or `visit2` and perform the `call_first` operation on the active variant member. This involves a second dimension to search: we have to explore the variant member space and match against the index variable to invoke `std::get<i>` and extract an lvalue to it.

Circle offers two idioms for this outer search: either use a multi ternary operator, or programmatically generate a switch.

```cpp
template<typename... types_t, typename... funcs_t>
auto visit1(const std::variant<types_t...>& variant, funcs_t&&... fs) {
  // Use a multi conditional operator and forward to call_first.
  return int...(sizeof...(types_t)) == variant.index() ...?
    call_first(std::get<int...>(variant), std::forward<funcs_t>(fs)...) :
    __builtin_unreachable();
}
```

`visit1` use the former technique. The `a` condition expression is the comparison of an integer sequence from 0 to `sizeof...(types_t) - 1` with the active variant index. Recall that `int...(count)` is a [pack index](https://github.com/seanbaxter/circle/blob/master/universal/README.md#pack-indices) operator. When this condition evaluates true, `call_first` is called with the active variant member, forwarding the entire parameter pack of function candidates.

```cpp
template<typename... types_t>
auto visit2(const std::variant<types_t...>& variant, auto&&... fs) {
  // Generate a switch and use a ...?? in each case.
  switch(variant.index()) {
    @meta for(int i : sizeof...(types_t)) {
      case i:
        return requires { fs(std::get<i>(variant)); } ...?? 
          fs(std::get<i>(variant)) :
          static_assert(@type_string(types_t...[i]));
    }
  }
}
```

`visit2` programmatically builds a switch. I like this solution because it doesn't involve any external functions, so we don't have to fuss with the `std::forward` mechanism, which is easy to mess up. Each switch case corresponds to a variant member, and uses constexpr multi conditional to invoke the first admissible function call.

`visit1` and `visit2` aren't strictly the same. The former uses [[expr.cond]](http://eel.is/c++draft/expr.cond) to convert each conditional subexpression's result object to a common type and returns that. `visit2` uses [return type placeholder deduction](https://eel.is/c++draft/dcl.spec.auto#general-8). Each of the non-discarded return statements must return the same type, or the program is ill-formed.