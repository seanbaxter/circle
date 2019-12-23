# Circle new macros

Build 82 of Circle features new macros. These are different and easier to use than the old macros, which have been removed from the compiler.

As with the old macros, there are two kinds of new macros:
1. **Statement macros**, which have an `@mvoid` return type.
1. **Expression macros**, which have an `@mauto` return type.

Macros are declared like ordinary functions. They undergo name lookup and overload resolution. You can even make macro templates, which undergo argument deduction, just like function templates.

The chief difference between macros and functions is that the macro definition is expanded into the calling scope every time it is called. Because macros are instantiated at each call, the body of the macro knows the scope into which it is being expanded, and can expose this scope to its own definition. A special form of name lookup will pass over the declarative region of the macro and begin searching at the inner-most enclosing non-meta scope.

