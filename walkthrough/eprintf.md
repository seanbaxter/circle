# Walkthrough 2: Evaluating expressions from text

[Walkthrough 1](functions.md) demonstrated injecting code from text. Here we flip the script, and try to extract meaning from text.

Let's write a printf-style function in the Circle style. Call it `eprintf`, for extended printf. Our novelty? We'll embed expressions right in the format specifier!

With printf, you need to escape the expression then evaluate it and pass it as a variadic argument:

```cpp
double x = 5;
printf("x = %f sqrt = %f exp = %f\n", x, sqrt(x), exp(x));
```

printf is way more consise than cout, but it's not very safe. It's not type safe, and it's easy to transpose the arguments relative to their escapes in the format specifier. Could we write a printer that allows embedding expressions right in the format specifier? Yes. All it takes is compile-time execution.

```cpp
double x = 5;
std::cout<< "x = {x} sqrt = {sqrt(x)} exp = {exp(x)}\n"_e;
```

This is one of the uses of eprintf. `operator""_e` is a user-defined literal that takes a string and invokes `esprintf`, which processes the format specifier, evaluates the arguments, converts the result objects to strings, and glues everything together. Right away something should occur to you: none of these can be functions. A function call would establish a function scope, and put the parser outside the declarative region in which `x` is declared. In other words, if it were a function, the definition of `esprintf` wouldn't be able to find `x`. 

We'll use Circle expression macros to implement this tricky operation. Expression macros, like statement macros, are called like functions (they undergo template argument deduction and overload resolution) but expand their definitions directly into the calling scope. Expression macros are declared with an `auto` return type. Statement macros are declared with a `void` return type. Expression macros may only have one non-meta statement, a _return-statement_ in which the argument expression is detached and inserted into the calling expression.

By implementing the `operator""_e` user-defined literal and `esprintf` as expression macros, we keep the declarations in the format specifier's embedded expression in scope.

## Simple eprintf

Our implementation strategy involves transforming the eprintf format specifier to a printf format specifier. Every set of braces gets stripped out and replaced by `%s`. Every braced expression gets evaluated, run through `std::to_string`, and passed as a variadic argument to `format`, which is a wrapper around [`vsnprintf`](https://en.cppreference.com/w/cpp/io/c/vfprintf).

The functions for transforming the format specifier are normal C++ functions. They're called at compile time, but don't need to be `constexpr`, because Circle is very permissive with respect to what can be executed at compile time.

[**eprintf1.cxx**](eprintf1.cxx)
```cpp
inline const char* parse_braces(const char* text) {
  const char* begin = text;

  while(char c = *text) {
    if('{' == c)
      return parse_braces(text + 1);
    else if('}' == c)
      return text + 1;
    else
      ++text;    
  }

  throw std::runtime_error("mismatched { } in parse_braces");
}


inline void transform_format(const char* fmt, std::string& fmt2, 
  std::vector<std::string>& names) {

  std::vector<char> text;
  while(char c = *fmt) {
    if('{' == c) {
      // Parse the contents of the braces.
      const char* end = parse_braces(fmt + 1);
      names.push_back(std::string(fmt + 1, end - 1));
      fmt = end;
      text.push_back('%');
      text.push_back('s');

    } else if('%' == c && '{' == fmt[1]) {
      // %{ is the way to include a { character in the format string.
      fmt += 2;
      text.push_back('{');

    } else {
      ++fmt;
      text.push_back(c);
    }
  }

  fmt2 = std::string(text.begin(), text.end());
}

@macro auto esprintf(const char* fmt) {
  // Process the input specifier. Remove {name} and replace with %s.
  // Store the names in the array.
  @meta std::vector<std::string> names;
  @meta std::string fmt2;
  @meta transform_format(fmt, fmt2, names);

  // Convert each name to an expression and from that to a string.
  // Pass to sprintf via format.
  return format(
    @string(fmt2.c_str()), 
    std::to_string(@expression(@pack_nontype(names))).c_str()...
  );
}

@macro auto eprintf(const char* fmt) {
  return std::cout<< esprintf(fmt);
}

@macro auto operator ""_e(const char* fmt, size_t len) {
  return esprintf(fmt);
}

int main() {
  double x = 5;
  std::cout<< "x = {x} sqrt = {sqrt(x)} exp = {exp(x)}\n"_e;

  return 0;
}
```
```
$ circle eprintf1.cxx 
$ ./eprintf1
x = 5.000000 sqrt = 2.236068 exp = 148.413159
```

We start by writing the format specifier with a `_e` literal suffix, which runs name lookup and finds the associated overloaded user-defined literal function. But in case, it's not a function, it's a macro! It receives the string literal's pointer and length as _compile-time arguments_. It passes the string literal to `esprintf`, which does the real work.

This macro calls `transform_format`, which strips out the braces, replaces them with `%s` escapes, and stores the text inside the braces in a vector of strings. We want to evaluate each expression string, stringify with `std::to_string`, and pass as a variadic argument to `format`.

This is super easy in Circle. We'll simply use [`@pack_nontype`](https://github.com/seanbaxter/circle/blob/master/packs/pack.md#pack_nontype) to expose all the strings in the `names` array as an unexpanded non-type parameter pack. In Circle, parameter packs don't need to be bound to template parameters. There are nearly a dozen operators for yielding parameter packs from different data structures; collectively they serve as a critical bridge between the imperative world of vectors and the functional world of variadic templates. Would you know how to call the variadic function `format` without `@pack_nontype`? It would be much much harder, if even possible.

`@pack_nontype` returns the elements in the `names` vector one at a time. We'll run those strings through `@expression`, which tokenizes, parses and injects them as expressions. Given our input text, this yields an lvalue double for the object reference `x` and prvalue doubles for the calls to `sqrt` and `exp`. 

The result object of each expression is then fed through `std::to_string`, which is an overload set of functions that converts arithmetic types to strings. Finally, we get a pointer to the string with the `c_str` member function. The pack expansion ellipsis `...` at the end of the argument performs the action on each member of the parameter pack, resulting in a list of function arguments for `format`.

This simple exercise demonstrates something critical: we can define a DSL (format specifiers are certainly a form of domain-specific language), process the DSL, then use Circle mechanisms to interface with the rest of the translation unit. `decltype(@expression(expr))`, for instance, would return the type of the embedded expression, which could serve as the starting point for type introspection and serialization. Since the scripting and compiled-language uses Circle use a common ABI, type system, declarative regions, and so on, there's little impedence mismatch between DSls implemented Circle and normal C++ code.

The beefier DSLs [Apex](https://github.com/seanbaxter/apex/blob/master/examples/autodiff.md), [RPN](https://github.com/seanbaxter/circle/blob/master/gems/rpn.md) and [peg-dsl](https://github.com/seanbaxter/circle/blob/master/peg_dsl/peg_dsl.md) all use `@expression` to gain knowledge about the expressions an objects referenced in their input strings. 

## Advanced eprintf

The simple eprintf function has a serious shortcoming: `std::to_string` is only defined to stringify arithmetic types. The real printf (when escaped properly) also prints strings. Circle, however, is a powerful language. We want to evolve eprintf to print darn near any type. For this, we'll dip into its type introspection features.

[**eprintf2.cxx**](eprintf2.cxx)
```cpp
struct vec3_t {
  double x, y, z;
};

enum class robot_t {
  T800,
  R2D2,
  RutgerHauer,
  Mechagodzilla,
  Bishop,
};

int main() {

  std::string s = "Hello world";
  robot_t robot = robot_t::Mechagodzilla;
  int array[] { 4, 5, 6 };
  vec3_t vec { 3.141, 6.626, 2.998 };

  std::cout<< "s = {s}\nrobot = {robot}\narray = {array}\nvec = {vec}\n"_e;

  std::cout<< "pair = {std::make_pair(robot, vec)}\n"_e;

  return 0;
}
```
```
$ circle eprintf2.cxx
$ ./eprintf2
s = Hello world
robot = Mechagodzilla
array = [ 4, 5, 6 ]
vec = { x : 3.141, y : 6.626, z : 2.998 }
pair = { first : Mechagodzilla, second : { x : 3.141, y : 6.626, z : 2.998 } }
```

There's a lot of value in this upgraded eprintf. It prints enumerator names for enums. It prints the elements of arrays and std::vector in bracketed lists. It prints std::map in braced key-value lists. We even use `std::make_pair` to synthesize a type, and eprintf nails that: the .first member is an enumerator name; the .second member is itself a class object, which gets printed out member-by-member.

The eprintf function doesn't know about any of our types, but it handles them correctly. Circle provides some type introspection services, and these are used to access enumerator and class member info.

```cpp
template<typename type_t>
const char* name_from_enum(type_t e) {
  static_assert(std::is_enum<type_t>::value);
  
  switch(e) {
    @meta for enum(type_t e2 : type_t) {
      // @enum_value is the i'th unique enumerator in type_t.
      // eg, circle, square, rhombus
      case e2:
        // @enum_name returns a string literal of the enumerator.
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

template<typename type_t>
void stream_simple(std::ostream& os, const type_t& obj) {

  if constexpr(std::is_enum<type_t>::value) {
    // For the simple stream, just write the enumerator name, not the 
    // enumeration type.
    if(const char* name = name_from_enum<type_t>(obj)) {
      // Write the enumerator name if the value maps to an enumerator.
      os<< name;
      
    } else {
      // Otherwise cast the enum to its underlying type and write that.
      os<< (typename std::underlying_type<type_t>::type)obj;
    }

  } else if constexpr(@is_class_template(type_t, std::basic_string)) {
    // For the simple case, stream the string without quotes. This is closer
    // to ordinary printf behavior.
    os<< obj;

  } else if constexpr(std::is_same<const char*, typename std::decay<type_t>::type>::value) {
    os<< obj;

  } else if constexpr(std::is_array<type_t>::value) {
    os<< '[';
    bool insert_comma = false;
    for(const auto& x : obj) {
      // Move to the next line and indent.
      if(insert_comma)
        os<< ',';
      os<< ' ';
      
      // Stream the element.
      stream_simple(os, x);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< " ]";    

  } else if constexpr(@is_class_template(type_t, std::vector)) {
    // Special treatment for std::vector. Output each element in a comma-
    // separated list in brackets.
    os<< '[';
    bool insert_comma = false;
    for(const auto& x : obj) {
      // Move to the next line and indent.
      if(insert_comma)
        os<< ',';
      os<< ' ';
      
      // Stream the element.
      stream_simple(os, x);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< " ]";

  } else if constexpr(@is_class_template(type_t, std::map)) {
    // Special treatment for std::map.
    os<< '{';
    bool insert_comma = false;
    for(const auto& x : obj) {
      if(insert_comma)
        os<< ",";
      os<< ' ';

      // stream the key.
      stream_simple(os, x.first);

      os<< " : ";

      // stream the value.
      stream_simple(os, x.second);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }  
    os<< " }";

  } else if constexpr(@is_class_template(type_t, std::optional)) {
    // For an optional member, either stream the value or stream "null".
    if(obj)
      stream_simple(os, *obj);
    else
      os<< "null";

  } else if constexpr(std::is_class<type_t>::value) {
    // For any other class, treat with circle's introspection.
    os<< '{';
    bool insert_comma = false;
    @meta for(size_t i = 0; i < @member_count(type_t); ++i) {
      if(insert_comma) 
        os<< ",";
      os<< ' ';

      // Stream the name of the member. The type will be prefixed before the
      // value.
      os<< @member_name(type_t, i)<< " : ";

      // Stream the value of the member.
      stream_simple(os, @member_ref(obj, i));

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< " }";

  } else {
    // For any non-class type, use the iostream overloads.
    os<< obj;
  }
}

template<typename type_t>
std::string cir_to_string(const type_t& obj) {
  std::ostringstream oss;
  stream_simple(oss, obj);
  return oss.str();
}
```

`cir_to_string` is the entry point for the new object stringifier. We'll allocate an `std::ostringstream` object to buffer our concatenations, and call into `stream_simple`, a recursive function that completely serializes an object.

`stream_simple` is itself a bunch of `if constexpr` cases on different aspects of the argument type. Enums are stringified with `name_from_enum`, which uses `@enum_name` to convert an enumerator to the string literal of its identifier. Arrays and std::vector are bracketed and printed in comma-separated lists. `std::optional` is serialized as its value (if set) and null otherwise. `std::map` and generic classes are printed as key-value pairs. In all cases, the `stream_simple` function is called recursively to serialize members in a collection. This gives us a JSON-like formatting that works for a large class object types.

