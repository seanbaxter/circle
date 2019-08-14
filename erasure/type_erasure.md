# Type erasure in Circle

Sy Brand wrote a Twitter [thread](https://twitter.com/TartanLlama/status/1159450865176129538) on "type erasure," a programming technique that I had never really thought about. In most languages, you write a base class with pure virtual functions that represents an interface. Then you write concrete classes which derive the interface and implement the virtual functions to do something class-specific.

Type erasure allows you to essentially rebind existing concrete classes to a new interface. This never took off with C++, because there's no way to automatically generate these bindings, even with C++ 20. You can manually write bindings or try to hack them together with preprocessor macros. But the innovative aspect of the "type erasure" pattern is lost among all the boilerplate of a manual implementation.

Sy Brand wrote an [example implementation](https://github.com/TartanLlama/typeclasses/blob/master/typeclass.hpp) that uses metaclasses and Reflection TS and builds with an experimental branch of Clang. Someone forwarded this to me and said that Circle ought to be able to do this. Well, Circle couldn't do it, but it's a very reasonable request. The goal is to make a compiler that's generally programmable, so I added a couple new mechanisms to the Circle compiler (build 51) and wrote my own type erasure implementation. 

I think it's a lot cleaner than the Reflection TS version, because the fragment injection mechanism is really ugly compared to Circle's meta statements.

To make this all work, I had to add two new mechanisms from whole cloth:

1. Member function introspection:
    * `@method_count` returns the number of non-static identifier-named member functions in the argument type.
    * `@method_type` returns the type of the i'th method as a pointer-to-member function.
    * `@method_name` returns the name of the i'th method as a string literal.
1. Programmable function declarations:
    To declare and initialize an object, just write `obj_type obj_name = obj_initializer`. It's a syntax that is easily programmable.  
    To declare and define a function is a lot harder to automate: `ret_type func_name(param-types-and-names)`. 
    If we don't know many how parameters there are when writing the source code, how do we produce a generic declaration? Well, introduce a new function declaration syntax:
    * `@func_decl` takes three arguments:
        1. A type describing the type of the function. This is be a function or function pointer type for non-member and static member functions. This is a pointer-to-member function for non-static member functions. (In the latter case, the record type may refer to any class, not just the one we're declaring the member function in.) For reflection purposes, we get this straight from `@method_type`.
        1. The name of the function as a string. We get this from `@method_name`.
        1. The identifier name of the function parameter pack. There's one function parameter for each parameter type in the function type, but it's not feasible to give them their own names. Instead, give them a function parameter pack name.

It's very convenient to declare a function from a function type. Not only does the function type define the return and parameter types, but it also incorporates the _noexcept-specifier_, and for non-static member functions the cv-qualifier (i.e. const and volatile qualifiers on the implicit `this` reference) and the ref-qualifier (the & and &&, allowing the function to selectively bind lvalue or xvalue objects). It's easy to define template specializations to selectively twiddle the noexcept, cv- and ref-qualifier parts of a type, as well as mutate the return and parameter types.

[**func_decl1.cxx**](func_decl1.cxx)
```cpp
#include <cstdio>
#include <cmath>

struct func_decl_t {
  const char* type;
  const char* name;
  const char* expr;
};

@meta func_decl_t func_decls[] {
  {
    "int(int, int)",
    "absdiff",
    "abs(args...[1] - args...[0])"
  },
  {
    "double(double)",
    "sq",
    "args...[0] * args...[0]"
  }
};

// Loop over the entries in func_decl_t at compile time.
@meta for(func_decl_t decl : func_decls) {

  // Declare a function for each entry.
  @func_decl(@type_id(decl.type), decl.name, args) {

    // Return the expression.
    return @expression(decl.expr);
  }
}

int main() {
  int x = absdiff(5, 7);
  printf("absdiff(5, 7) -> %d\n", x);

  double y = sq(3.14159);
  printf("sq(3.14158) -> %f\n", y);

  return 0;
}
```
```
$ circle func_decl1.cxx
$ ./func_decl1
absdiff(5, 7) -> 2
sq(3.14158) -> 9.869588
```

`@func_decl` is demonstrated by taking a function type, function name and function body by string and converting them into a proper C++ function. The third argument specifies the identifier name of the function parameter pack, which is why the string definitions access the parameters as `args...[0]` and `args...[1]`. In real practice, these would most likely be forwards to other functions.

[**func_decl2.cxx**](func_decl2.cxx)
```cpp
#include <cstdio>

template<typename interface>
struct my_base_t {
  // Loop over each method in interface_t.
  @meta for(int i = 0; i < @method_count(interface); ++i) {
    @meta printf("Injecting %s: %s\n", 
      @method_name(interface, i), 
      @type_name(@method_type(interface, i))
    );

    // Declare a pure virtual function with the same name and signature.
    virtual @func_decl(@method_type(interface, i), @method_name(interface, i), args) = 0;
  }
};

struct my_interface_t {
  void print(const char* text);
  bool save(const char* filename, const char* data);
  void close();
};


int main() {
  typedef my_base_t<my_interface_t> my_base;

  // Print all the method names and types:
  @meta for(int i = 0; i < @method_count(my_base); ++i) {
    @meta printf("Found %s: %s\n", 
      @method_name(my_base, i), 
      @type_name(@method_type(my_base, i))
    );
  }

  return 0;
}
```
```
$ circle func_decl2.cxx
Defining pure virtual print
Defining pure virtual save
Defining pure virtual close
Found print: void(my_base_t<my_interface_t>::*)(const char*)
Found save: bool(my_base_t<my_interface_t>::*)(const char*, const char*)
Found close: void(my_base_t<my_interface_t>::*)()
```

The `@func_decl` mechanism works hand-in-hand with method introspection. Here we define an interface type that declares three member functions. This type, `my_interface_t` might never be instantiated; it rather serves as a contract with my_base_t, which is a type with the same methods as virtual functions.

We meta for over the methods like we would non-static data members or enumerators, and use `@func_decl` on the `@method_type` and `@method_name` to declare the pure virtual function. Use `virtual` to set the function as a virtual function (since that's not part of the function type) and `= 0` to make it a pure/abstract function (that's not part of the type, either).

Injecting a pointer-to-member function type creates a non-static member function. The record type in the pointer-to-member is ignored, which makes it easy to introspect from an interface and inject into another class. In the example abvoe, method pointers to `my_interface_t` are used to inject declarations for `my_base_t`.

## Type erasure with Circle

My implementation of type erasure differs from Sy Brand's in a couple of ways:
1. A compiler-generated virtual table on model_t is used rather than a manually-constructed one. This means that model_t is only 8 bytes for the vptr, rather than 8 bytes per function in the interface, plus storage for additional pointers for copy/move construction. C++ provided virtual table construction (at a considerable cost to the compiler vendor), so it's good sense to use that mechanism rather than roll our own.
1. There's no special `move_clone_` support in `model_t`. If you want to move, just std::move the `var_t` object itself; this will move-assign/move-construct the `unique_ptr` that holds the storage for the var.
1. Sy Brand's wrapper class is called "storage". I call it var_t to emphasize that it is similar to polymorphic var types in javascript and other dynamic languages. "storage" sounds like an implementation detail, whereas "var" sounds like a user-facing class.

We want a single type to access polymorphic behavior without requiring a specific kind of inheritance. Let's code for this interface:

[**type_erasure.cxx**](type_erasure.cxx)
```cpp
////////////////////////////////////////////////////////////////////////////////
// The var_t class template is specialized to include all member functions in
// my_interface. It makes forwarding calls from these to the virtual 
// functions in model_t.

struct my_interface {
  // List interface methods here. They don't have to be virtual, because this
  // class never actually gets inherited!
  void print(const char* text);
};

// Print the text in forward order.
struct forward_t {
  void print(const char* text) {
    puts(text);
  }
};

// Print the text in reverse order.
struct reverse_t {
  void print(const char* text) {
    int len = strlen(text);
    for(int i = 0; i < len; ++i)
      putchar(text[len - 1 - i]);
    putchar('\n');
  }
};

// Print the text with caps.
struct allcaps_t {
  void print(const char* text) {
    while(char c = *text++)
      putchar(toupper(c));
    putchar('\n');
  }
};

// The typedef helps emphasize that we have a single type that encompasses
// multiple impl types that aren't related by inheritance.
typedef var_t<my_interface> obj_t;

int main() {

  // Construct an object a.
  obj_t a = obj_t::construct<allcaps_t>();
  a.print("Hello a");

  // Copy-construc a to get b.
  obj_t b = a; 
  b.print("Hello b");

  // Copy-assign a to get c.
  obj_t c;
  c = b;
  c.print("Hello c");

  // Create a forward object.
  obj_t d = obj_t::construct<forward_t>();
  d.print("Hello d");

  // Create a reverse object.
  obj_t e = obj_t::construct<reverse_t>();
  e.print("Hello e");

  return 0;
}
```
```
$ circle type_erasure.cxx 
$ ./type_erasure 
HELLO A
HELLO B
HELLO C
Hello d
e olleH
```

We have an interface `my_interface` that declares a single method, `print`, although you can declare as many methods as you'd like. Unlike `std::function` or `std::bind`, this type erasure paradigm allows any number of interface methods.

There are three classes that implement member functions `print`, but they have no inheritance relationship with `my_interface`. What we want is to use Circle metaprogramming to generate a specialization `var_t<my_interface>` that binds the `print` member functions from the concrete types to a set of virtual functions that match those declared in the interface. The `var_t` type also manages the lifetime of the concrete object, providing us with a single type interface for the methods on different classes that are unrelated by inheritance.

[**type_erasure.cxx**](type_erasure.cxx)
```cpp
// A Circle implementation of the type erasure tactic implemented here:
// https://github.com/TartanLlama/typeclasses/blob/master/typeclass.hpp

#include <memory>
#include <vector>
#include <cstdlib>

// model_t is the base class for impl_t. impl_t has the storage for the 
// object of type_t. model_t has a virtual dtor to trigger impl_t's dtor.
// model_t has a virtual clone function to copy-construct an instance of 
// impl_t into heap memory, which is returned via unique_ptr. model_t has
// a pure virtual function for each method in the interface class typeclass.
template<typename typeclass>
struct model_t {
  virtual ~model_t() { }

  virtual std::unique_ptr<model_t> clone() = 0;

  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    // Declare a pure virtual function for each interface method.
    virtual @func_decl(@method_type(typeclass, i), @method_name(typeclass, i), args) = 0;
  }
};

template<typename typeclass, typename type_t>
struct impl_t : public model_t<typeclass> {

  // Construct the embedded concrete type.
  template<typename... args_t>
  impl_t(args_t&&... args) : concrete(std::forward<args_t>(args)...) { }

  std::unique_ptr<model_t<typeclass> > clone() override {
    // Copy-construct a new instance of impl_t on the heap.
    return std::make_unique<impl_t>(concrete);
  }
 
  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    // Declare an override function with the same signature as the pure virtual
    // function in model_t.
    @func_decl(@method_type(typeclass, i), @method_name(typeclass, i), args) override {
      // Forward to the correspondingly-named member function in type_t.
      // std::forward<@method_params(typeclass, i)>(args)... works too.
      return concrete.@(__func__)(std::forward<decltype(args)>(args)...);
    }
  }

  // Our actual data.
  type_t concrete;
};

////////////////////////////////////////////////////////////////////////////////
// var_t is an 8-byte type that serves as the common wrapper for the 
// type-erasure model_t. It implements move 

template<typename typeclass>
struct var_t {
  // Default initializer creates an empty var_t.
  var_t() = default;

  // Allow initialization from a unique_ptr.
  var_t(std::unique_ptr<model_t<typeclass> >&& model) : 
    model(std::move(model)) { }

  // Move ctor/assign by default.
  var_t(var_t&&) = default;
  var_t& operator=(var_t&&) = default;

  // Call clone for copy ctor/assign.
  var_t(const var_t& rhs) {
    if(rhs)
      model = rhs.model->clone();
  }

  var_t& operator=(const var_t& rhs) {
    model.reset();
    if(rhs)
      model = rhs.model->clone();
    return *this;
  }

  // A virtual dtor triggers the dtor in the impl.
  virtual ~var_t() { }

  // The preferred initializer for a var_t. This constructs an impl_t of
  // type_t on the heap, and stores the pointer in a new var_t.
  template<typename type_t, typename... args_t>
  static var_t construct(args_t&&... args) {
    return var_t(std::make_unique<impl_t<typeclass, type_t> >(
      std::forward<args_t>(args)...
    ));
  }

  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {
    // Declare a non-virtual forwarding function for each interface method.
    @func_decl(@method_type(typeclass, i), @method_name(typeclass, i), args) {
      // Forward to the model's virtual function.
      return model->@(__func__)(std::forward<decltype(args)>(args)...);
    }
  }

  explicit operator bool() const {
    return (bool)model;
  }

  // This is actually a unique_ptr to an impl type. We store a pointer to 
  // the base type and rely on model_t's virtual dtor to free the object.
  std::unique_ptr<model_t<typeclass> > model;
}; 

```

The type erasure implementation is barely more complicated than the two `@func_decl` examples. There are three classes at work:

1. `var_t` is the user-facing wrapper. This holds a `unique_ptr` to `model_t`, which manages the actual data and binding. Move construction/assignment is handled by move-semantics on the `unique_ptr`. Copy constructiont/assignment is handled by a `clone` function on the `model_t` object.
1. `model_t` is specialized on the interface type, called `typeclass` for consistency with Sy Brand's code. This type reflects the interface's methods as pure virtual functions. It also declares a `clone` virtual function to invoke the copy constructor on the concrete type. It serves as a base class for the `impl_t` class template, so we also add a virtual dtor to allow the derived type to destruct itself.
1. `impl_t` derives `model_t`, and is additionally specialized on a concrete type like `forward_t` or `allcaps_t`. It reflects on the interface methods, creating virtual function overrides which forward the function arguments to the corresponding member function on the concrete type.

Note that inside a function we can use `__func__` to access the function name. This is a compile-time constant string literal defined for all functions in C++. Placing it in the dynamic-name operator `@()` like `@(__func__)` converts the string name to an identifier.

Circle's support for type erasure was coded up in a hurry. `@func_decl` was something I came up with after perhaps an hour of consideration, and I'm sure I'll expand it moving forward. But I feel this mechanism, when combined with Circle's existing meta statements, already has a big leg up over the metaclasses implementation:

```cpp
  -> __fragment { 
     dispatch_table[idx] = reinterpret_cast<void(*)()>(&impl::unqualid(func));
  
```
The Circle version doesn't `reinterpret_cast` function pointers and build its own vtable. That's all done by the compiler, as it should be.

```cpp
  -> __fragment class X { public:
      static typename(ret) unqualid(func) (base& mod, ->params) {
          return static_cast<X&>(mod).ct.unqualid(func)(unqualid(...params));
      }
  };
```
Circle doesn't have such an inscrutable way of programmatically declaring functions. `@func_decl` is unfamiliar because it's new, but it's just got three composable pieces: a function type, a function name and a parameters name. The metaclasses syntax is a mess of symbols. In Circle, the parameters are accessed as a normal parameter pack with the `...` behind the declaration. In the metaclasses code the `...` come in front? And there's a `->` in front of the parameters? 

It's tricky to programmatically declare fuctions, which is why Circle declares the entire function from a pointer-to-member function type. We already have techniques for manipulating types (and Circle adds a bunch of imperative `@mtype`-related methods on top of the C++11 features). Use these mechanisms to prepare your function type, then punch the function out with a `@func_decl`. There's no need to introduce so much new syntax for fine-grained declarations.

## Specifying core and optional methods

It doesn't take much work to write a smarter type erasure system. The most obvious improvement is to support both core/required and extended/optional methods in the interface. In a real world situation, you'd want to make the interface as rich as possible, but still allow binding to implementations that don't implement all the methods. 

```cpp
struct my_interface {
  enum class required {
    print         // Only the print method is required.
  };

  // Required methods:
  void print(const char* text);

  // Optional methods:
  void save(const char* filename, const char* access);
};
```

Write your methods like normal, but annotate the required ones by naming them in the `required` enum. If you omit this enum, all methods are interpreted as optional. Concrete types must implement all required methods for the program to compile. How does the caller know which methods are actually implemented, since it only has access to the generic `var_t` wrapper? We'll generate a `has_{func-name}` function for each method, which returns true if implemented and false if not.

That is, we'll expect `var_t<my_interface>` to look like this:
```cpp
struct var_t {
  bool has_print() const;
  void print(const char* text);

  bool has_save() const;
  void save(const char* filename, const char* access);

  // Data members here...
};
```

The user should check `has_save` before calling `save`, since that is an optional method. If the user goes ahead and calls an unimplemented optional method, an exception with a useful error message will be thrown.

[**type_erasure2.cxx**](type_erasure2.cxx)
```cpp
// model_t is the base class for impl_t. impl_t has the storage for the 
// object of type_t. model_t has a virtual dtor to trigger impl_t's dtor.
// model_t has a virtual clone function to copy-construct an instance of 
// impl_t into heap memory, which is returned via unique_ptr. model_t has
// a pure virtual function for each method in the interface class typeclass.
template<typename typeclass>
struct model_t {
  virtual ~model_t() { }

  virtual std::unique_ptr<model_t> clone() = 0;

  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    @meta std::string func_name = @method_name(typeclass, i);

    // Declare a "has_" function.
    virtual bool @(format("has_%s", func_name.c_str()))() const = 0;

    // Declare a pure virtual function for each interface method.
    virtual @func_decl(@method_type(typeclass, i), func_name, args) = 0;
  }
};

template<typename typeclass, typename type_t>
struct impl_t : public model_t<typeclass> {

  // Construct the embedded concrete type.
  template<typename... args_t>
  impl_t(args_t&&... args) : concrete(std::forward<args_t>(args)...) { }

  std::unique_ptr<model_t<typeclass> > clone() override {
    // Copy-construct a new instance of impl_t on the heap.
    return std::make_unique<impl_t>(concrete);
  }
 
  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    @meta std::string func_name = @method_name(typeclass, i);

    @meta bool is_valid = @sfinae(
      std::declval<type_t>().@(func_name)(
        std::declval<@method_params(typeclass, i)>()...
      )
    );

    // Implement the has_XXX function.
    bool @(format("has_%s", func_name.c_str()))() const override {
      return is_valid;
    }

    // Declare an override function with the same signature as the pure virtual
    // function in model_t.
    @func_decl(@method_type(typeclass, i), func_name, args) override {

      @meta if(is_valid || @sfinae(typeclass::required::@(__func__))) {
        // Forward to the correspondingly-named member function in type_t.
        return concrete.@(__func__)(std::forward<decltype(args)>(args)...);

      } else {

        // We could also call __cxa_pure_virtual or std::terminate here.
        throw std::runtime_error(@string(format("%s::%s not implemented", 
          @type_name(type_t), __func__
        )));
      }
    }
  }

  // Our actual data.
  type_t concrete;
};

```

We've added a `has_{func-name}` prue virtual to `model_t` and an override function to `impl_t`. To test if the method is actually implemented on the concrete type, the Circle `@sfinae` extension is used to simulate a function call in an unevaluated context. `std::declval` is used to synthesize an instance of the concrete type and each of the function arguments. If the member function name lookup succeeds and overload resolution chooses a candidate, then the whole expression will substitute correctly and yield true. If there's any error during substitution, the `@sfinae` extension will suppress the diagnostic and yield false.

The method implementation in `impl_t` will attempt to call the corresponding function on the concrete object if such a call will succeed (`is_valid` is true) or if it's a required function. To check the required flag, evaluate the `typeclass::required::@(__func__)` expression in a `@sfinae` context. If this name lookup succeeds, the extension returns true.

If the function call is both optional and invalid, we'll throw a C++ exception that combines the name of the concrete type and the function name into the error message.

**Note on meta objects:**
> To save typing we stored the function name into a meta std::string called `func_name`. Notice that I'm using `func_name`s value only in the initializer for `is_valid` and for the two function declarations. I'm not using it from _inside_ a function definition.
>
> Why? Well, the definitions of functions in a templated context (such as function templates or member functions of class templates) aren't instantiated when the surrounding class is instantiated. Rather, they are instantiated when used. But when they're used, the meta loops in the class definition have already been executed, and their objects have been created and destroyed. `i`, `func_name` and `is_valid` aren't alive when the definitions that reference them are instantiated into code, so we can't access their values. How come we can refer to `is_valid` and `i` from inside these function definitions then? When a function in a templated context is declared, the Circle compiler _memoizes_ all the surrounding meta objects. For arithmetic and enum types, it records the type and value of the object. For pointer and non-scalar types, it only records the type. That string object cannot be used from inside a function definition, since the compiler didn't memoize its value. (In general, we can't memoize types with non-trivial ctors and dtors, although I will be adding support for literal types and probably common STL types like vectors of memoizable types and for strings.) 
>
> We have the same consideration when parsing member function definitions for non-templated classes. The definitions are parsed only after the rest of the body of the class has been processed, making the class complete. At that point, the meta for loops have run, their objects created and destroyed, and all that remains for the function definition to access are memoized versions of the objects.
>
> Using `__func__` from inside the function definitions yields a perfectly suitable string literal that doesn't have these memoization concerns.

```cpp
struct my_interface {
  enum class required {
    print         // Only the print method is required.
  };

  void print(const char* text);
  void save(const char* filename, const char* access);
};

// Print the text in forward order.
struct forward_t {
  void print(const char* text) {
    puts(text);
  }

  void save(const char* filename, const char* access) {
    puts("forward_t::save called");
  }
};

// Print the text in reverse order.
struct reverse_t {
  void print(const char* text) {
    int len = strlen(text);
    for(int i = 0; i < len; ++i)
      putchar(text[len - 1 - i]);
    putchar('\n');
  }
};

// Print the text with caps.
struct allcaps_t {
  void print(const char* text) {
    while(char c = *text++)
      putchar(toupper(c));
    putchar('\n');
  }
};

// The typedef helps emphasize that we have a single type that encompasses
// multiple impl types that aren't related by inheritance.
typedef var_t<my_interface> obj_t;

int main() {

  // Construct an object a.
  obj_t a = obj_t::construct<allcaps_t>();
  a.print("Hello a");

  // Copy-construct a to get b.
  obj_t b = a; 
  b.print("Hello b");

  if(b.has_save())
    b.save("my.save", "w");

  // Copy-assign a to get c.
  obj_t c;
  c = b;
  c.print("Hello c");

  // Create a forward object.
  obj_t d = obj_t::construct<forward_t>();
  d.print("Hello d");

  // save works, because forward_t implements it.
  d.save("foo.save", "w");

  // Create a reverse object.
  obj_t e = obj_t::construct<reverse_t>();
  e.print("Hello e");

  // Throws:
  // terminate called after throwing an instance of 'std::runtime_error'
  //   what():  reverse_t::save not implemented
  e.save("bar.save", "w");

  return 0;
}
```
```
$ circle type_erasure2.cxx
$ ./type_erasure2
HELLO A
HELLO B
HELLO C
Hello d
forward_t::save called
e olleH
terminate called after throwing an instance of 'std::runtime_error'
  what():  reverse_t::save not implemented
Aborted (core dumped)
```

Only `forward_t` implements the optional `save` method. We guard against calling `save` on object `b` with the `has_save` member function. Later on, we call `save` on a `reverse_t` object, and that throws the exception.

```cpp
    // Define a has_XXX member function.
    bool @(format("has_%s", @method_name(typeclass, i)))() const {
      @meta if(@sfinae(typeclass::required::@(__func__)))
        return true;
      else
        return model->@(__func__)();
    }
```
It's worth noting that the `var_t::has_{func-name}` implementation returns the true constant if the function is required (which implies valid). This allows the compiler backend to inline through this guard, making it performance penalty-free to use before any required interface call.

