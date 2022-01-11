# Circle variant 

[**variant code here**](variant.hxx)

This is a Circle implementation of C++20's [`std::variant`](http://eel.is/c++draft/variant) class. The goal of this exercise isn't about providing a faster-compiling variant, although it it that. Like my [mdspan implementation](https://github.com/seanbaxter/mdspan#mdspan-circle), working through variant is an opportunity to extend the language so that writing such advanced code no longer poses a challenge.

The libstdc++ variant is [very scary](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/variant). If the veteran C++ programmers who implemented this had so much trouble, what hope is there for the rest of is?

This new variant is a simple transliteration from standardese. It leverages a bunch of existing Circle features:

* [Data member pack declarations](https://github.com/seanbaxter/mdspan#data-member-pack-declarations)
* [Pack subscript operator](https://github.com/seanbaxter/circle/blob/master/universal/README.md#static-subscripts-and-slices) `...[]`
* [Pack index operator](https://github.com/seanbaxter/circle/blob/master/universal/README.md#pack-indices) `int...` and integer pack generator `int...(N)`
* [Multi-conditional operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#multi-conditional---) `...?:`
* [Constexpr conditiontal operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-conditional--) `??:`
* [Member type traits](https://github.com/seanbaxter/circle/blob/master/imperative/README.md#type-traits) `.template`, `.type_args` and `.string` 
* Pack `static_assert`

But that wasn't enough for a clean variant. To address pain points discovered in variant, I implemented two more major features:

1. `__preferred_copy_init` and `__preferred_assignment` provide the intelligence for the converting constructor and converting assignment operator. They perform overload resolution given an expression and a collection of types, and indicate the type that has the best viable construction or assignment, or failure if there is no viable operation, or multiple best-viable operations.
2. `__visit` and `__visit_r` is a multi-dimensional, single-expression visitor operator. It implements `std::visit` on any number of variant arguments in one line.

## Member pack unions, constructor and destructor.

The [mdspan implementation](https://github.com/seanbaxter/mdspan#mdspan-circle) introduced data member [data member pack declarations](https://github.com/seanbaxter/mdspan#data-member-pack-declarations). In that application, non-static data members of a class were declared variadically. In this exercise, _variant members_ and their subobject initializers are declared variadically.

[**variant.hxx**](variant.hxx)
```cpp
template<class... Types>
class variant {
  static constexpr bool trivially_destructible = 
    (... && std::is_trivially_destructible_v<Types>);

  union {
    Types ...m;
  };

  uint8_t _index = variant_npos;

public:
  constexpr variant() 
  noexcept(std::is_nothrow_default_constructible_v<Types...[0]>) 
  requires(std::is_default_constructible_v<Types...[0]>) : 
    m...[0](), _index(0) { }

  constexpr ~variant() requires(trivially_destructible) = default;
  constexpr ~variant() requires(!trivially_destructible) { reset(); }

  constexpr void reset() noexcept {
    if(_index != variant_npos) {
      _index == int... ...? m.~Types() : __builtin_unreachable();
      _index = variant_npos;        // set to valueless by exception.
    }
  }
};
```

Use the `...member-name` _declarator_ syntax inside a _union-specifier_ to declare a pack of variant members. We no longer have to use inheritance to implement `std::variant` -- it's all one self-contained class.

Since we have an unnamed union with potentially non-trivially constructible members, we should specify a subobject initializer for the first variant member in the variant class's default constructor. Circle features a pack subscript operator `...[N]`, which is used to specify:
* The _noexcept-specifier_ for construction of the first variant member,
* The _requires-clause_ constraint that deactivates the constructor if the first variant member isn't default constructible, and
* Subobject access to the first variant member in the unnamed unions.

### Conditionally trivial destructor.

C++20 permits conditionally trivial destructors, in which multiple destructors are declared, marked by different _requires-clauses_, one of which may be marked `= default`, to create a trivial destructor. When the class is made complete (when the parser hits the closing brace of its definition, or when it's instantiated into a template specialization), the destructor constraints are evaluated and exactly one destructor candidate is chosen.

```cpp
  constexpr ~variant() requires(trivially_destructible) = default;
  constexpr ~variant() requires(!trivially_destructible) { reset(); }
```

If all variant members have trivial destructors, we default the destructor's definition. This will suppress any code generation for the destructor.

### Multi-conditional reset.

If any variant member has a non-trivial destructor, like an `std::string` that should deallocate dynamic memory, the user-defined destructor is defined, which delegates to the `reset` member function.

```cpp
  constexpr void reset() noexcept {
    if(_index != variant_npos) {
      _index == int... ...? m.~Types() : __builtin_unreachable();
      _index = variant_npos;        // set to valueless by exception.
    }
  }
```

`reset` is where the choice of a member pack union starts paying off. We want to switch over all valid indices (that is, all indices exception `variant_npos`, which indicates the _valueless by exception_ state) and call the destructor of the corresponding variant member. Using the [multi-conditional operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#multi-conditional---) `...?:`, this is performed with a single expression:
```cpp
      _index == int... ...? m.~Types() : __builtin_unreachable();
```

`_index` is our scalar index of the currently-set variant member. `int...` is a pack expression, which when substituted as part of a pack expansion expression `...`, yields back the current index of the expansion as an index. This pack expression infers the pack size from other pack expressions in the same expansion. 

`_index == int...` is a pack of comparison expressions. For a variant with four alternatives, it'll expand out like this:
```
(_index == 0, _index == 1, _index == 2, _index == 3)
```

Of course, only one of these can be true. Each of these comparisons serves as the left-hand side of a conditional operator. We want to effect an operation like this:
```cpp
_index == 0 ? m...[0].~Types...[0]() :
_index == 1 ? m...[1].~Types...[1]() :
_index == 2 ? m...[2].~Types...[2]() :
_index == 3 ? m...[3].~Types...[3]() :
              __builtin_unreachable();
```

The multi-conditional operator `...?:` will expand out the left, middle and right-hand operands to generate precisely that.

```cpp
      _index == int... ...? m.~Types() : __builtin_unreachable();
```

* The left-hand operand is the predicate, which yields true when we're substituting on the index corresponding to the active variant member. 
* The center operand calls the _pseudo-destructor_ on the member pack declaration `m`. The type of the _pseudo-destructor_ is the `variant` template parameter `Types`. During pack expansion, the i'th variant member and i'th `Types` element are substituted in concert, forming a valid desturctor operation.
* The right-hand operand is `__builtin_unreachable()`. This is compiler lingo for noting that a branch of execution is unreachable. It allows the optimizer to employ _strength reduction_ passes to improve code quality. We're basically telling the compiler that we've accounted for all code paths, even the unreachable ones.

## Copy construction.

```cpp
  // Copy ctors.
  constexpr variant(const variant& w)
  requires(trivially_copy_constructible) = default;

  constexpr variant(const variant& w)
  noexcept(nothrow_copy_constructible) 
  requires(copy_constructible && !trivially_copy_constructible) {
    if(!w.valueless_by_exception()) {
      int... == w._index ...? 
        (void)new(&m) Types(w. ...m) : 
        __builtin_unreachable();
      _index = w._index;
    }
  }
```

If all variant members are trivially copy constructible, a _requires_clause_ selects the default definition, so that the emitted copy constructor is a simple `memcpy`. Otherwise, the user-defined copy constructor is selected.

The multi-conditional operator `...?:` does all the work, as it will do over and over in this variant implementation. The predicate for the conditional is a comparison of the current pack index with the active variant member of the rhs. When that predicate is true, we need _placement-new_ to invoke the copy constructor of the variant member that's going to become active. The syntax is `new(pointer) Type(arguments)`. But we have to do this as a pack, with a bunch of separate pack elements:
* The pointer to the active member is a pack, `&m`. For each element of the pack expansion instantiated, this will yield the address the corresponding variant member in the union.
* The type of the active member, is a pack, `Types`. 
* The initializer exprsession of the active member is a pack, `w. ...m`. `w` is a parameter of a _dependent type_, `variant`. Although this is also the type of _current instantiation_, C++ rules require the compiler to defer name lookup until instantiation, only then will subobjects of dependent base classes be known. The `...` token that precedes the _member-id_ indicates that, upon name lookup, the member will be a pack declaration. We can't currently write `w....m`, because greedy lexing rules would tokenized that is `w... .m`, which is nonsense. If you forget to put the `...` disambiguating token before the member name, the compiler will remind you when you instantiate the template.


