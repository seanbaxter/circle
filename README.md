# Circle
The C++ Automation Language  
2023
Sean Baxter

Download [here](https://www.circle-lang.org/)

Follow me on Twitter [@seanbax](https://www.twitter.com/seanbax) for compiler updates.

## New Circle

* [New Circle notes](new-circle/README.md)

New Circle is a major transformation of the Circle compiler, intended as a response to recent [successor language announcements](https://accu.org/journals/overload/30/172/teodorescu/). It focuses on a novel [fine-grained versioning mechanism](new-circle/README.md#versioning-with-feature-pragmas) that allows the language to **fix defects** and make the language **safer** and **more productive** while maintaining **100% compatibility** with existing code assets.

New Circle is the richest C++ compiler yet. Try out:

* [choice types](new-circle/README.md#choice),
* [pattern matching](new-circle/README.md#pattern-matching),
* [interfaces and impls](new-circle/README.md#interface),
* [language type erasure](new-circle/README.md#language-type-erasure),
* [_as-expressions_](new-circle/README.md#as) for safer conversions,
* [a modern declaration syntax](new-circle/README.md#new_decl_syntax) with `fn` and `var` keywords to make clearer, less ambiguous declarations,
* [a simpler syntax for binary expressions](new-circle/README.md#simpler_precedence), greatly reducing the liklihood of bugs caused by confusing operator precedences,
* [a `forward` keyword](new-circle/README.md#forward) to take the complexity and bugginess out of forwarding references,
* [safer initializer lists](new-circle/README.md#safer_initializer_list), which address ambiguities when calling std::initializer_list constructors and non-std::initializer_list constructors,
* [lifting lambdas](new-circle/README.md#overload-sets-as-arguments) to pass overload sets as function arguments,
* [nine kinds of template parameters](new-circle/README.md#template-parameter-kinds) to make templates far more comprehensive,
* [reflection traits](new-circle/README.md#reflection-traits) to access packs of information about class types, enum types, function types, class specializations, and so on,
* [pack traits](new-circle/README.md#pack-traits) for pack-transforming algorithms, like sort, unique, count, erase, difference, intersection, and so on.

New Circle describes a path for evolving C++ to meet the needs of institutional users. The versioning mechanism that accommodated the development of the features above will also accommodate research into critically important areas like memory safety. Rather than insisting on a one-size-fit's-all approach to language development, project leads can opt into collections of features that best target their projects' needs.

## Old docs. May be out of date. Refer to [New Circle](new-circle/README.md) for fresh information.

* [Circle implementations of Standard Library classes](stdlib#circle-implementations-of-standard-library-classes)  
    * [Member packs and `std::tuple`](tuple#circle-tuple)  
    * [Member packs and `std::variant`](variant#circle-variant)  
    * [Member packs and `mdspan`](https://github.com/seanbaxter/mdspan/tree/circle#mdspan-circle)  
* [CUDA](cuda/README.md)  
* [Pattern Matching](pattern/README.md)  
* [Circle Imperative Arguments](imperative/README.md)  
* [Member traits](member-traits/README.md)  
* [Circle conditionals](conditional/README.md)  
* [Universal member access](universal/README.md)  
* [Circle reflection and typed enums](reflection/README.md)  
* [Circle C++ for shaders](https://www.github.com/seanbaxter/shaders)  
* [List comprehensions, slices, ranges, for-expressions, functional folds and expansion expressions](comprehension/README.md)  
* [File @embed and a compile-time design dilemma](embed/embed.md)  
* [The Circle format library](fmt/fmt.md)  
* [Compile-time regular expressions](regex/regex.md)  
