# Circle tuple

Browse implementation [**tuple.hxx**](tuple.hxx).

This is a Circle implementation of C++20's [`std::tuple`](http://eel.is/c++draft/tuple) class.

Also note the Circle [variant](../variant#circle-viriant) and Circle [mdspan](https://github.com/seanbaxter/mdspan#mdspan-circle) implementations.

## Contents.

* [First-class tuple support](#first-class-tuple-support)
* [Data member packs](#data-member-packs)
* [Mapping types to indices](#mapping-types-to-indices)
* [Deduced forward references](#deduced-forward-references)
* [Tuple cat](#tuple-cat)

## First-class tuple support.

As documented [here](../universal#tuple-subscripts-and-slices), 



[**access.cxx**](access.cxx)
```cpp
#include "tuple.hxx"
#include <array>
#include <iostream>

int main() {
  circle::tuple<int, double, const char*> x1(
    100, 3.14, "Hello circle::tuple"
  );

  // Print out by subscript.
  std::cout<< "Print by subscript:\n";
  std::cout<< "  0: "<< x1.[0]<< "\n";
  std::cout<< "  1: "<< x1.[1]<< "\n";
  std::cout<< "  2: "<< x1.[2]<< "\n";

  std::tuple<short, float, std::string> x2(
    50, 1.618f, "Hello std::tuple"
  );

  // Print out by slice.
  std::cout<< "Print by slice:\n";
  std::cout<< "  " + int....string + ": "<< x2.[:]<< "\n" ...;

  std::pair<const char*, long> x3(
    "A pair's string",
    42
  );
  std::cout<< "Works with pairs:\n";
  std::cout<< "  " + int....string + ": "<< x3.[:]<< "\n" ...;

  std::cout<< "Even works with builtin arrays:\n";
  int primes[] { 2, 3, 5, 7, 11 };
  std::cout<< "  " + int....string + ": "<< primes.[:]<< "\n" ...;
}
```
```
$ circle access.cxx && ./access
Print by subscript:
  0: 100
  1: 3.14
  2: Hello circle::tuple
Print by slice:
  0: 50
  1: 1.618
  2: Hello std::tuple
Works with pairs:
  0: A pair's string
  1: 42
Even works with builtin arrays:
  0: 2
  1: 3
  2: 5
  3: 7
  4: 11
```

## Data member packs.

Member pack declarations are used in the Circle mdspan for [_partially-static storage_](https://github.com/seanbaxter/mdspan#data-member-pack-declarations) and [inside unions](../variant#circle-variant) in the Circle variant.

[**pack1.cxx**](pack1.cxx) 
```cpp
#include <iostream>
#include <utility>

template<typename... Ts>
struct tuple {
  // Declare a parameter pack of data members with ...<name> declarator-id.
  [[no_unique_address]] Ts ...m;

  // Declare default, copy and move constructors.
  tuple() : m()... { }
  tuple(const tuple&) = default;
  tuple(tuple&&) = default;

  // Converting constructor. Note the ... after the m pack subobject init.
  template<typename... Ts2>
  requires((... && std::is_constructible_v<Ts, Ts2&&>))
  tuple(Ts2&&... x) : m(x)... { }

  // Specialize a single element. Subobject-initialize that one element,
  // and default construct the rest of them.
  template<size_t I, typename T>
  tuple(std::in_place_index_t<I>, T&& x) :
    m...[I](x), m()... { }

  // Use pack subscript ...[I] to access pack data members.
  template<int I>
  Ts...[I]& get() {
    return m...[I];
  }

  template<typename T>
  T& get() {
    // The requested type must appear exactly once in the tuple.
    static_assert(1 == (... + (T == Ts)));
    constexpr size_t I = T == Ts ...?? int... : -1;
    return m...[I];
  }
};

struct empty1_t { };
struct empty2_t { };
struct empty3_t { };

// Members of the same type do not alias under no_unique_address rules.
static_assert(3 == sizeof(tuple<empty1_t, empty1_t, empty1_t>));

// Members of different types do alias under no_unique_address rules.
static_assert(1 == sizeof(tuple<empty1_t, empty2_t, empty3_t>));

int main() {
  // Use the converting constructor to create a tuple.
  tuple<int, double, const char*> x(10, 3.14, "Hello tuple");
  std::cout<< x.get<0>()<< " "<< x.get<1>()<< " "<< x.get<2>()<< "\n";

  // Initialize only member 1 with 100.0.
  tuple<int, double, float> y(std::in_place_index<1>, 100);
  std::cout<< y.get<0>()<< " "<< y.get<1>()<< " "<< y.get<2>()<< "\n";

  // Print the int element of x and the double element of y.
  std::cout<< x.get<int>()<< " "<< y.get<double>()<< "\n";
}
```

The heart of the tuple implementation is the data member pack declaration `Ts ...m`. At definition, `m` is a parameter pack declaration, and any expression naming it is a pack expression. At instantiation, concrete data members are created, with names `m0`, `m1`, `m2` and so on. These names are necessary for reflection and useful for error reporting.

The member pack is given the [`[[no_unique_address]]`](https://en.cppreference.com/w/cpp/language/attributes/no_unique_address), which helps compress the data structure by allowing empty members with different types alias to the same offset within the class. This is equivalent to the [empty base optimization](https://en.cppreference.com/w/cpp/language/ebo), but in a more useful form. It is prohibited to directly inherit from multiple base classes of the same type, but it is perfectly fine to expand a member pack with multiple elements of the same type. This member pack-driven tuple does away with the indexing wrappers that appear in every ISO C++ tuple implementation.

To initialize member packs subobjects in bulk, use `m(args)...`, the same syntax that you'd use to initialize base class pack subobjects. However, Circle also lets you initialize specific data members with the [pack subscript operator](https://github.com/seanbaxter/circle/blob/master/universal/README.md#pack-subscripts-and-slices) `m...[I](args)`. Initialize as many subscripted elements as you'd like, and then initialize the remainder of the pack with the bulk initializer `m(args)...`. 

## Mapping types to indices.

Pack subscript makes implementing _getter_ functions trivial. Take the element index as a template parameter and use `...[I]` on the data member. `std::tuple` and `std::variant` also support getters on type template parameters, if that type parameter occurs exactly once in the container.

```cpp
  template<typename T>
  T& get() {
    // The requested type must appear exactly once in the tuple.
    static_assert(1 == (... + (T == Ts)));
    constexpr size_t I = T == Ts ...?? int... : -1;
    return m...[I];
  }
```

We'll use an additive fold-expression to enforce the "occurs once" mandate. The operand of the fold is simply `T == Ts`, which compares each type in `Ts` to the get parameter `T`. If they're the same, the subexpression is `true`, which gets promoted 1, and summed up.

To map this unique type `T` to its index within `Ts`, use [constexpr multi-conditional operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-multi-conditional---) `??:`. The left-hand operand is a pack expression predicate for the conditional. Substitution progress until it finds a pack element that evaluates `true`, and then it substitutes the corresponding pack element of the center operand, which it yields as the result of the expression. Our center operand is the [pack index operator](https://github.com/seanbaxter/circle/blob/master/universal/README.md#pack-indices) `int...`. This operator is itself a parameter pack expression, which yields the current index of expansion during substitution. Therefore, if `T == Ts` was true on pack element 3, `int...` yields 3 when substituted, which becomes the value of the index used for subscripting the data member pack `m`.

## Deduced forward references.

The big invention added for implementing `std::tuple` is the _deduced forward reference_. This extension to overload resolution eliminates the need for multiple function overloads that only differ on the const, volatile or reference qualifiers of their parameters. Both the forwarding type of the function parameter and constituent parts of its _type-id_ are deduced.

> _parameter-declaration_ : _type-id_

The syntax is simple: write a forwarding reference parameter declaration, like `T&& u` where `T` is a template parameter of that function, a colon, and a _type-id_ for the type of the function parameter itself. Ordinary forwarding references are unconstrained, in that `T` can be deduced to be any type. But deduced forwarding parameters are set by the right-hand type, and function arguments must be converted to that. Effectively, the const, volatile and reference qualifiers are deduced from the argument, and applied to the right-hand type.

This is still, at heart, a forwarding reference. During overload resolution, type type of an lvalue argument is replaced by its own lvalue reference, and deduction takes place from there.

As an example:

```cpp
template<typename T, typename... Args>
void f(T&& u : std::tuple<Args...>);
```

`T` can deduced to any of eight possible types, allowing this one function to take the place of up to eight separate overloads:

```cpp
template<typename... Args>
void f(std::tuple<Args...>& u);

template<typename... Args>
void f(const std::tuple<Args...>& u);

template<typename... Args>
void f(volatile std::tuple<Args...>& u);

template<typename... Args>
void f(const volatile std::tuple<Args...>& u);

template<typename... Args>
void f(std::tuple<Args...>&& u);

template<typename... Args>
void f(const std::tuple<Args...>&& u);

template<typename... Args>
void f(volatile std::tuple<Args...>&& u);

template<typename... Args>
void f(const volatile std::tuple<Args...>&& u);
```

This sample uses Circle reflection to confirm that deduced forward references do properly handle const and non-const lvalue and xvalue argument types.

[**deduce.cxx**](deduce.cxx) 
```cpp
#include <iostream>
#include <utility>

template<typename... Ts>
struct tuple {
  Ts ...m;
};

template<typename T1>
void f1(T1&& x) {
  std::cout<< "  f1: "<< T1.string<< "\n";
} 

template<typename T2, typename... Args>
void f2(T2&& y : tuple<Args...>) {
  std::cout<< "  f2: "<< T2.string<<" | Args: ";
  std::cout<< Args.string<< " "...;
  std::cout<< "\n";
}

struct derived_t : tuple<int, char, void*> { };

int main() {
  derived_t d1;
  const derived_t d2;

  std::cout<< "lvalue:\n";
  f1(d1);
  f2(d1);

  std::cout<< "const lvalue:\n";
  f1(d2);
  f2(d2);

  std::cout<< "xvalue:\n";
  f1(std::move(d1));
  f2(std::move(d1));

  std::cout<< "const xvalue:\n";
  f1(std::move(d2));
  f2(std::move(d2));
}
```
```
$ circle deduce.cxx && ./deduce
lvalue:
  f1: derived_t&
  f2: tuple<int, char, void*>& | Args: int char void* 
const lvalue:
  f1: const derived_t&
  f2: const tuple<int, char, void*>& | Args: int char void* 
xvalue:
  f1: derived_t
  f2: tuple<int, char, void*> | Args: int char void* 
const xvalue:
  f1: const derived_t
  f2: const tuple<int, char, void*> | Args: int char void* 
```

[P2481R0 - Forwarding reference to specific type/template](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2481r0.html) highlights a number of problems with general unconstrained forwarding refernces. [P0847R7 - Deducing 'this'](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0847r7.html) specifically considers the ["shadowing problem"](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2481r0.html#the-shadowing-mitigation-private-inheritance-problem) in which deducing a derived type in an explicit 'this' function can make members of the base class inaccessible. 

[**self.cxx**](self.cxx) 
```cpp
#include <iostream>
#include <utility>

struct B1 {
  template<typename Self>
  auto&& get0(this Self&& self) {
    // Error: ambiguous declarations found from id-expression 'i'.
    return std::forward<Self>(self).i;
  }

  template<typename Self>
  auto&& get1(this Self&& self) {
    // P00847R7: mitigate against shadowing by copy-cvref.
    return ((__copy_cvref(Self, B1)&&)self).i;
  }

  template<typename Self>
  auto&& get2(this Self&& self : B1) {
    // Circle deduced forward reference uses a normal forward.
    return std::forward<Self>(self).i;
  }

  int i;
};

struct B2 {
  int i;
};

struct D : B1, B2 { };

int main() {
  D d;

  // Uncomment this for ambiguous declaration error.
  // int x0 = d.get0();

  // Works with explicit upcast to B1.
  int x1 = d.get1();

  // Works with deduced forward reference.
  int x2 = d.get2();
}
```

The basic problem is that forwarding references, including those deducing "this" forwarding references, bind to the type of the argument passed, not to the type of the argument of interest. For general robustness, the user must upcast back to the argument type it wants, but in a way that preserves the const, volatile and reference qualifiers of the forwarding reference. The Circle method, as in `get2`, deduces the template parameter to a type `B1`, so that it can be used directly as the `std::forward` template argument.

[tuple.cnstr](https://eel.is/c++draft/tuple#lib:tuple,constructor______)
```cpp
template<class U1, class U2> constexpr explicit(see below) tuple(pair<U1, U2>& u);
template<class U1, class U2> constexpr explicit(see below) tuple(const pair<U1, U2>& u);
template<class U1, class U2> constexpr explicit(see below) tuple(pair<U1, U2>&& u);
template<class U1, class U2> constexpr explicit(see below) tuple(const pair<U1, U2>&& u);
```

The C++ Standard defines `std::tuple` constructors and operators in sets of four: const and non-const, lvalue and rvalue parameter-taking. The deduced forwarding reference lets us write one overload and get functionality for all four:

[**tuple.hxx**](tuple.hxx)
```cpp
  // Conversion constructor from std::pair.
  template<class T, class U1, class U2>
  requires(
    sizeof...(Types) == 2 &&
    std::is_constructible_v<Types...[0], __copy_cvref(T&&, U1)> &&
    std::is_constructible_v<Types...[1], __copy_cvref(T&&, U2)>
  )
  constexpr explicit(
    !std::is_convertible_v<__copy_cvref(T&&, U1), Types...[0]> || 
    !std::is_convertible_v<__copy_cvref(T&&, U2), Types...[1]>
  )
  tuple(T&& u : std::pair<U1, U2>) :
    m((get<int...>(std::forward<T>(u))))... { }
```

`T` is the forwarding reference of the pair parameter. It can assume on of these four types, to match each of the four overloads for the tuple constructor:

1. `std::pair&`
2. `const std::pair&`
3. `std::pair`
4. `const std::pair`

The constraint and _explicit-specifier_ require determining the type of the pair element members as if they were accessed with a `get` around a `forward`. We can accomplish this with the new Circle compiler builtin `__copy_cvref`, which copies the const, volatile and reference qualifiers from the first operand to the type of the second operand.

## Tuple cat.

The most difficult `std::tuple` function to implement using ordinary C++ is be [`tuple_cat`](
https://eel.is/c++draft/tuple#lib:tuple_cat):

```cpp
template<class... Tuples>
constexpr tuple<CTypes...> tuple_cat(Tuples&&... tpls);
```

The libstdc++ implementation uses [recursive partial template specialization](https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/libstdc%2B%2B-v3/include/std/tuple#L1651). But this should be an easy operation. It's really just a double for loop: the outer loop visits the parameters in `tpls`, and the inner loop visits the tuple elements in each parameter.

[**tuple.hxx**](tuple.hxx)
```cpp
template<class... Tuples>
constexpr tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      auto N : Ti.remove_reference.tuple_size =>
        get<int...(N)>(std::forward<Ti>(tpls...[i]))...
  };
}
```

[Circle Imperative Arguments](https://github.com/seanbaxter/circle/tree/master/imperative#readme) provides control flow within template argument lists, function argument lists and initializer lists. [_argument-for_](https://github.com/seanbaxter/circle/tree/master/imperative#argument-for) is the tool of choice here. To form the function's return type, we _argument-for_ inside the tuple's _template-argument-list_. For each parameter `Ti` in `Tuples`, expand the pack `Ti.remove_reference.tuple_elements`. This is a usage of Circle member traits, which rewrites C++ type traits using a member-like syntax for clarity. `tuple_elements` yields a parameter pack by querying `std::tuple_size` for the pack size and probing `std::tuple_elements` for each pack member.

The function's body is just the return statement with a hulked out initializer list. This uses the two-declaration version of _argument-for_, where the first declaration `i` is the index of iteration, and the second declaration `typename Ti` holds the current type in the collection `Tuples`. For each type parameter, we use _argument-let_ to declare a value `N`, which is set with the number of tuple elements. Finally, there's a pack expansion expression that forwards the `i`th function parameter `tpls...[i]` using its forwarding reference `Ti` into a `get` function, specialized on each integer between 0 and `N` - 1. That final line essentially blows out a tuple into its elements.

```cpp
template<class... Tuples>
constexpr tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat2(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      std::forward<Ti>(tpls...[i]).[:] ...
  };
}
```

But this function is _even easier_ with Circle's [first-class tuple support](#first-class-tuple-support). We don't have to form a call to `get` to destructure each function parameter into the _initializer-list_. We can simply use the tuple slice operator `.[:]` directly on each function parameter, and pack expand that.

Because the `circle::tuple` class registers itself with the `std::tuple_size` extension point, the Circle frontend can infer its tuple size and access its constituent elements with an ADL call to `get`.
