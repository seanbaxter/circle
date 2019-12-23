# File @embed and a compile-time design dilemma

`@embed` is a new extension in Circle that takes a file path, loads the named file, and yields a constant array with the file's data. Like a string literal, this may initialize an array object, or it may be decayed to a pointer.

[ThePHD](https://twitter.com/thephantomderp) has been leading the charge for an embed extension with [P1040](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1040r4.html). He wrote up his experience with standardization and Circle in an amusing rant [here](https://thephd.github.io/full-circle-embed), with follow-up performance information here[](https://thephd.github.io/embed-the-details).

## @embed usage

[**embed1.cxx**](embed1.cxx)
```cpp
#include <cstdio>

const char* filename = "test_binary.data";
const int data[] = @embed(int, filename);

@meta printf("data has %zu bytes\n", sizeof(data));

int main() {
  // Use it or lose it.
  for(int x : data)
    printf("%d\n", x);

  return 0;
}
```

To use `@embed`, specify the inner type of the array and the path of the file to load. An optional third parameter takes the number of elements to read from the file. If the third parameter is not provided, the bounds on the array is inferred from the length of the file. The filename must be an expression that can be evaluated at compile time, because this operation is strictly compile time--it loads the file during semantic analysis and embeds its contents into a constant array, right in the binary output.

## Compile-time programming with @embed

Circle has an integrated interpreter that will execute any code at compile time. Rather than naming specific files to load, we can use POSIX functions to scrape filenames from a given directory, and programmatically load those files with `@embed`. To assist the user, a pointer into this constant data and its length are stored in an `std::map` which is dynamically accessible at runtime.

```cpp
// Make stat available.
#define __USE_EXTERN_INLINES
#include <sys/stat.h>

// Support directory ops.
#include <dirent.h>

#include <string>
#include <vector>
#include <map>
#include <cstdio>

inline std::vector<std::string> get_dir_filenames(std::string dirname) {
  std::vector<std::string> filenames;

  DIR* dir = opendir(dirname.c_str());

  // Loop over all entities in the directory.
  while(dirent* e = readdir(dir)) {
    // Concatenate the dirname and filename.
    std::string filename = dirname + "/" + e->d_name;

    // Match regular files.
    struct stat statbuf;
    if(0 == stat(filename.c_str(), &statbuf) && S_ISREG(statbuf.st_mode))
      filenames.push_back(std::move(filename));
  }

  closedir(dir);
  return filenames;
}

// Relate filenames to file contents.
@meta auto filenames = get_dir_filenames("resources");
@meta puts(filenames[:].c_str())...;

// Populate a runtime std::map relating filenames to contents.
// The @embed operation is performed at compile time, but the map is 
// constructed at runtime.
// The map, being a global, can also be used at compile time by the 
// interpreter. Since the compiler knows the initializer, it JIT-constructs the
// map when ODR-used by the interpreter. The files have already been loaded 
// by this point, and indexed through a deduplicator.

// We can also declare a meta std::map. This has the advantage of not 
// generating any wasteful code, and it may still be used at runtime, but 
// the indices must be compile-time strings.

struct range_t {
  const char* begin;
  size_t size;
};

template<size_t count>
range_t make_range(const char(&array)[count]) {
  return range_t { array, count };
}

std::map<std::string, range_t> file_map = {
  std::make_pair<std::string, range_t>(
    @string(@pack_nontype(filenames)),
    make_range(@embed(char, @pack_nontype(filenames)))
  )...
};

// Print the files loaded at compile time.
@meta+ for(auto& item : file_map) {
  printf("%s (%zu bytes): %.*s\n", item.first.c_str(), 
    item.second.size, item.second.size, item.second.begin);
}


int main() {
  for(auto& item : file_map) {
    printf("%s (%zu bytes): %.*s\n", item.first.c_str(), 
      item.second.size, item.second.size, item.second.begin);
  }
  return 0;
}
```
```
$ circle embed_loader.cxx 
resources/test2.txt
resources/test3.txt
resources/test1.txt
resources/test1.txt (22 bytes): This is test1.txt file
resources/test2.txt (22 bytes): This is test2.txt file
resources/test3.txt (22 bytes): This is test3.txt file

$ ./embed_loader 
resources/test1.txt (22 bytes): This is test1.txt file
resources/test2.txt (22 bytes): This is test2.txt file
resources/test3.txt (22 bytes): This is test3.txt file
```

`get_dir_filenames` is an ordinary function. It uses no compile-time tricks. It takes a directory name and returns the names of all normal files in that directory. However, we execute this funtion at compile time to get the listing of the `resources` directory in the build tree.

```cpp
@meta auto filenames = get_dir_filenames("resources");
@meta puts(filenames[:].c_str())...;
```

After querying the directory, a slice expand statement prints the name of each file to load as a diagnostic.

```cpp
std::map<std::string, range_t> file_map = {
  std::make_pair<std::string, range_t>(
    @string(@pack_nontype(filenames)),
    make_range(@embed(char, @pack_nontype(filenames)))
  )...
};
```

This pithy map initializer constructs an `std::map` in the global namespace. It's not a meta object, but an ordinary object that can be used at runtime. The `@pack_nontype` extension converts the `std::vector<std::string> filenames` object to a static parameter pack of strings, and `@string` converts each of those strings to a string literal (i.e. a constant char array) that is portable to runtime. After pack expansion, this map initializer is equivalent to:

```cpp
std::map<std::string, range_t> file_map = {
  std::make_pair<std::string, range_t>(
    "resources/test2.txt",
    make_range(@embed(char, "resources/test2.txt"))
  ),
  std::make_pair<std::string, range_t>(
    "resources/test3.txt",
    make_range(@embed(char, "resources/test3.txt"))
  ),
  std::make_pair<std::string, range_t>(
    "resources/test1.txt",
    make_range(@embed(char, "resources/test1.txt"))
  )
};
```

With just one declaration we've essentially created a virtual file system to a collection of pre-loaded files.

## The dilemma

This is expressive, compact code. It does something very specific in very few lines. But the construction of it was quite deliberate. Why did I choose to write a function `get_dir_filenames` that only returns the array of filenames in the specified directory? Why not also put the `@embed` usage in that function, and have it insert the node pointing to the file's data in the `std::map` as well?

```cpp
inline void embed_dir_files(std::string dirname, 
  std::map<std::string, range_t>& file_map) {

  std::vector<std::string> filenames;

  DIR* dir = opendir(dirname.c_str());

  // Loop over all entities in the directory.
  while(dirent* e = readdir(dir)) {
    // Concatenate the dirname and filename.
    std::string filename = dirname + "/" + e->d_name;

    // Match regular files.
    struct stat statbuf;
    if(0 == stat(filename.c_str(), &statbuf) && S_ISREG(statbuf.st_mode)) {
      // Embed the file.
      file_map.insert({
        filename, 
        make_range(@embed(char, filename))
      });
    }
  }

  closedir(dir);
}
```

This function has a clear logic--loop over all files in a directory and embed their contents into the provided map. It seems like it _ought_ to work. But it _won't_ work. The `@embed` extension returns a constant array. The extent of the embedded file is encoded in the return type. Because `@embed` has a data-dependent type, it must be invoked with a file path that is known at _definition_ or _substitution_. Although a function ought to be able to access the compile-time features of Circle when executed through the interpreter, it would be impossible to even compile this function, because of the type-dependent nature of `@embed`.

The limitations of `@embed` forced me to structure code in a way that, if not exactly awkward, is at least sensitive to the design of compile-time extensions. Rather than serving my needs, the Circle extensions require me to serve theirs.

### Definition and deferred contexts

What we have are _two_ compile-time contexts: the _definition context_ and the _deferred context_. The former context is when arguments are known at definition; this is the context of template arguments. The latter context is when only the type of arguments are known at definition, but their values will be provided during execution at compile time through the integrated interpreter.

Circle has dozens of new extensions, and it's my desire to make them available, when sensible, to both the definition and deferred contexts.

### String extents

Type-dependence afflicts most of the Circle extensions. All extensions returning strings yield constant arrays rather than `const char*` objects:

* `@enum_name`
* `@member_name`
* `@member_decl_string`
* `@type_string`

As currently implemented, these extensions must be used in definition contexts, fixing the extent in the array return types. Could we define variants that perform pointer decay and return `const char*` objects when called with non-constant arguments? We could, but then the semantics of the extension change subtly based on the availability of arguments.

An implicit pointer decay for a string is probably an acceptable trade for increased flexibility. But what do we do with `@embed`? Files aren't generally null-terminated. If we decay the result object to a pointer when `@embed` is called with non-constant arguments (and in a function executed by the interpreter at compile time), how do we retrieve the length of the pointed-at data? We'd need to change the return type to return a tuple that includes the pointer and the size. Or return the extent through an out parameter. Or make a second extension to access the size of the embedding.

### Expression injection

Consider the `@expression` extension, which allows injection of an expression from text. When the text is known at definition, the result object of the extension is inferred by parsing the text and producing an expression AST. But what if the provided text is just a function parameter? We'd be able to inject the text when the containing function is executed by the interpreter, but we wouldn't be able to compile the function, due to the type-dependence of `@expression`. 

What if we added a deferred context variant of `@expression` that fixes the type of the result object?

```cpp
double x = sin(@expression(double, my_text));
```

We know what the type of the result object is, because it's indicated by the first parameter. It no longer gets inferred from the input text `my_text`, which is critical because in a deferred context the value of that argument isn't known until execution.

### Circle macros

Circle macros are like functions. They have similar definitions. They undergo ordinary name lookup, overload resolution, and in the case of macro templates, argument deduction. They can decay to macro pointers, be stored in standard containers, then pulled out and invoked indirectly. They're just like functions. But unlike functions, the compiler parses Circle macros from their tokens every time they are textually called in the source.

This effectively gives us function definitions that operate in a definition context rather than the usual deferred context. Function parameters bind to values that are known at definition, so type-dependent extensions can be called with macro parameter-dependent arguments. 

Macros, being parsed from tokens whenever called, also allow access to the scope of call site. Macros can deposit real declarations which fall through layers of meta scopes and embed themselves into the inner-most enclosing real scope.

Macros as currently defined have two different properties that make them useful:
1. Definition-time availability of parameters lets us use type-dependent Circle extensions.
1. Access to the calling scope allows deposition of real declarations.

It's a categorical oddity that I've been employing macros for two separate reasons. Really, only the second property is a good use of the macro. Choosing a macro instead of a function for the first reason is a failure of language design: Circle ought to have deferred context versions of its type-dependent extensions so that we can use them from normal functions.

## A way forward

Circle has carved out a namespace by reserving the `@` character to prefix its many new extensions. Currently, all these extensions work in the definition context. The types of the result objects are inferred from the values of the arguments.

The plan moving forward is to use `@@` for the _deferred context_ namespace. These variant extensions have non-dependent or user-specified return types.

* `@@enum_name` returns a `const char*` to the name of an enumerator rather than a constant character array.
* `@@expression` copy-initializes a specified type given the injected text.
* `@@embed` returns a tuple with a pointer to the embedded text along with the number of elements embedded.
* `@@array` consumes a range of data, turns it into a binary literal, and returns a pointer to the result object. The extent of the array is already known from the parameter.

### @mtype

The Circle type `@mtype` is a pointer-sized opaque value that refers to any type. It supports the equivalence and relational operators, so that it can be sorted and uniquified using ordinary algorithms, executed in the interpreter.

[**tuple3.cxx**](https://github.com/seanbaxter/long_demo/blob/master/tuple3.cxx)
```cpp
#include <vector>
#include <algorithm>
#include <iostream>

template<typename value_t>
void stable_unique(std::vector<value_t>& vec) {
  auto begin = vec.begin();
  auto end = begin;
  for(value_t& value : vec) {
    if(end == std::find(begin, end, value))
      *end++ = std::move(value);
  }
  vec.resize(end - begin);
}

template<typename... types_t>
struct tuple_t {
  types_t @(int...) ...;
};

template<typename... types_t>
struct unique_tuple_t {
  @meta std::vector<@mtype> types { @dynamic_type(types_t)... };
  @meta stable_unique(types);
  typedef tuple_t<@pack_type(types)...> type_t;
};

template<typename... types_t>
using unique_tuple = typename unique_tuple_t<types_t...>::type_t;

// Turn these 7 elements into 4 unique ones.
typedef unique_tuple<int, double, char*, double, char*, float> my_tuple;

@meta std::cout<< @member_decl_strings(my_tuple)<< "\n" ...;

int main() {
  return 0;
}
```

`unique_tuple_t` specializes a vector of `@mtype` and initializes with `@mtype` objects for each of the tuple parameter pack elements. Because the member functions of `std::vector` have inline linkage, the resulting specializations aren't sent to the LLVM backend, which doesn't support `@mtype` operations. Those functions, along with `stable_unique`, are executed by the Circle interpreter, which does support these `@mtype` operations.

The existence of `@mtype` provides a mechanism for enabling deferred context versions of introspection extensions.

* `@@member_count` takes an `@mtype` and returns the count of non-static public data members. The specified type isn't resolved at definition--but rather when the enclosing function is executed by the interpreter.

* `@@member_type` takes an `@mtype` to identify the class type and an integer to select the non-static data member. The result object is another `@mtype`, which can be returned from a function, received in a definition context, and translated into the contained type with `@static_type`.

* `@@type_id` takes injects a string and yields not a type, but an `@mtype` that encapsulates the named type.

### Deferred lookup

Some Circle extensions perform name lookup from text, including `@expression` and `@type_id`. The deferred context versions of these extensions perform name lookup starting with the inner-most enclosing scope being defined, rather than the scope of the enclosing function. Functions are executed at compile time to assist code generation, and the code currently being generated is the scope of interest for dynamic name lookup. The scope of the function being executed has already been processed, and is locked in place. Modifying name lookup for `@@expression` allows functions using this extension to inject code from text and evaluate it in the scope still being defined. This is crucial for implementing tooling like [text formatting](https://github.com/seanbaxter/circle/blob/master/fmt/fmt.md).

## Deferred @@embed

```cpp
inline void embed_dir_files(std::string dirname, 
  std::map<std::string, range_t>& file_map) {

  std::vector<std::string> filenames;

  DIR* dir = opendir(dirname.c_str());

  // Loop over all entities in the directory.
  while(dirent* e = readdir(dir)) {
    // Concatenate the dirname and filename.
    std::string filename = dirname + "/" + e->d_name;

    // Match regular files.
    struct stat statbuf;
    if(0 == stat(filename.c_str(), &statbuf) && S_ISREG(statbuf.st_mode)) {
      // Embed the file.
      const char* data = @@embed(char, filename);
      size_t size = @@embed_size(data);
      file_map.insert({ filename, range_t { data, size } });
    }
  }

  closedir(dir);
}
```

This is one potential design for deferred `@embed`. `@@embed`, when executed, yields a pointer rather than a sized array. `@@embed_size`, when executed, resolves the binary from the pointer and returns the length of the array. This two-extension design resolves the type dependence of `@embed`. 
