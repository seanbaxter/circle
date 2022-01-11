# Circle variant 

[**variant code here**](variant.hxx)

This is a Circle implementation of C++20's [`std::variant` class](http://eel.is/c++draft/variant). The goal of this exercise isn't as much to provide a faster-compiling variant (although it is!), but to improve the language so that implementing this sophisticated class isn't a struggle.

The libstdc++ variant is [very scary](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/std/variant). If the veteran C++ programmers who implemented this had so much trouble, what hope is there for the rest of is?

Like my [mdspan implementation](https://github.com/seanbaxter/mdspan#mdspan-circle), this variant class is an opportunity to augment the Circle language with features to relieve programming pain. 

This new variant is a simple transliteration from standardese. It leverages a bunch of existing Circle features:

* [Data member pack declarations](https://github.com/seanbaxter/mdspan#data-member-pack-declarations)
* [Pack subscript operator](https://github.com/seanbaxter/circle/blob/master/universal/README.md#static-subscripts-and-slices) `...[]`
* [Pack index operator](https://github.com/seanbaxter/circle/blob/master/universal/README.md#pack-indices) `int...` and integer pack generator `int...(N)`
* [Multi-conditional operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#multi-conditional---) `...?:`
* [Constexpr conditiontal operator](https://github.com/seanbaxter/circle/blob/master/conditional/README.md#constexpr-conditional--) `??:`
* [Member type traits](https://github.com/seanbaxter/circle/blob/master/imperative/README.md#type-traits) `.template`, `.type_args` and `.string` 
* Pack `static_assert`

But that wasn't enough. To address pain points discovered in variant, I implemented two more major features:
1. `__preferred_copy_init` and `__preferred_assignment` provide the intelligence for the converting constructor and converting assignment operator. They perform overload resolution given an expression and a collection of types, and indicate the type that has the best viable construction or assignment, or failure if there is no viable operation, or multiple best-viable operations.
2. `__visit` and `__visit_r` is a multi-dimensional, single-expression visitor operator. It implements `std::visit` on any number of variant arguments in one line.

