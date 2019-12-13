# @embed and @array

## Literal types

C++17 introduces the concept of a "literal type," which allows certain kinds of classes to be ported between compile-time and runtime contexts implicitly. Technically this category consists of all scalar types (arithmetic, enum and pointer types), arrays, and literal classes, which are basically classes with only other literal types as data members. 

Being a literal type is a necessary but not sufficient condition for compile-time to runtime portability. We also have to be able to resolve mapping pointers (to functions and objects and into arrays and strings) from their compile-time targets to runtime targets, which involves translating binary data through various stores (maintaining all the objects with static storage duration, or all the functions, or all the string literals) and resolving handles to declarations. 

The LLVM output of these constants is not pure binary data, but has pointers to module-level objects, functions and constants mixed in. When the application is started, the dynamic linker resolves these addresses and stores the constant data, achieving a kind of labored portability between the compile-time and runtime domains.

To assist in porting types that aren't portable because they aren't literal types or have troublesome pointers, Circle introduces a few keywords. Pass the operator some compile-time data and it returns an expression that's resolved for runtime access. 

The most obvious and useful example is `@string`, which takes a pointer-to-character or `std::string`, reads out the data at compile time, and yields a const character array lvalue. (In effect a string literal, although there's no actual textual quotation, and implicit concatenation in the preprocessor and things like that won't work, since this operation happens during definition or instantiation.) The compiler gives you a runtime backing for information that's known at compile time, but you have to explicitly ask for it. This is the same way static reflection works--the compiler knows everything about every type, but only encodes that information into the output assembly when you ask for it with an introspection operator like `@member_name`.

## @array

Long ago I hacked up an `@array` operator which ports compile-time data into a const array prvalue, suitable for assignment into an array declaration. Pass it a pointer and a count, and it would yield a braced initializer with the contents of the input data converted from binary back into semantic objects. You could use the meta features to open a file handle, `fread` some data in, then `@array` it into a new global objects. Easy!

But I revisited it and it was a bit slow. It was, after all, breaking apart the input data into semantic objects, filling them one-by-one into a braced initializer, and exposing them to a laborious pointer-porting process during code generation. So I wrote a binary initializer node, which just copies the input data into a `std::vector<char>` and sends that straight to [`ConsantDataArray::getRaw`](https://llvm.org/doxygen/classllvm_1_1ConstantDataArray.html#a9e5ceea3bf75c4de560cf77e95b30a64) during code generation. After an hour of struggling with GEP, it worked. This is the fastest possible code path. Now `@array` takes no time. But still--loading a file from `FILE*` in the interpreter and turning it into a constant array with `@array` was still pretty slow. What gives?

## @embed 

[thephd](https://twitter.com/thephantomderp) messaged me on Twitter [(@seanbax)](https://twitter.com/seanbax) about loading a file during source translation and yielding a constant array. I sent him a sample using `@array`, then really got to examine its performance. I made the change described above, turning `@array` into a utility that just copies bits instead of running the gauntlet of C++ semantic analysis. But how to explain the continued poor performance?

I volunteered to write an `@embed` keyword, not aware that he had been running his own [`std::embed` proposal](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1040r4.html) for the past year and a half. But I immediately hit on the same idea as him, because it's a self-explanatory, narrowly-scoped, useful feature that should be [obvious](https://thephd.github.io/full-circle-embed) to anyone wanting to embed data from a file into a translation unit.

```cpp
@embed(type-id, filename, [count])
```

Provide `@embed` with the inner type of the array (i.e. `int` yields an `int[]` array), the filename as a `const char*`, `std::string` or `std::string_view`, and an optional count (in elements, not in bytes). If a count is provided and the file has insufficient bytes, an error is issued. If it has more than enough bytes, those extra bytes are ignored. If the count is not specified, it is inferred from the file length. If the file's length is not an integer multiple of the inner type size, an error is issued. Ordinary stuff!

[**gen_file.cxx**](gen_file.cxx)
```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>

const int size = 12'500'000;

int main() {
  std::vector<int> data(size);
  for(int i = 0; i < size; ++i)
    data[i] = 1000 * i + i;

  FILE* f = fopen("test_binary.data", "w");
  fwrite(data.data(), sizeof(int), data.size(), f);
  fclose(f);

  return 0;
}
```

I generated a 50MB binary file of ints. Now to embed it. (Drumroll.)

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

I load the binary file with `@embed` and assign it into a global variable. I print the size of the array at compile time as a diagnostic. I print the array contents from `main`, to prevent the LLVM optimizer from getting rid of it.

```
$ time circle embed1.cxx
data has 50000000 bytes

real  0m0.468s
user  0m0.332s
sys 0m0.136s

$ ls -al embed1
-rwxr-xr-x 1 sean sean 50008232 Dec 13 12:26 embed1

$ circle embed1.cxx -filetype=ll -console -O0 | more
; ModuleID = 'embed1.cxx'
source_filename = "embed1.cxx"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@data = internal constant [12500000 x i32] [i32 0, i32 1001, i32 2002, i32 3003, i32 4004, i32 5005, i32 6
006, i32 7007, i32 8008, i32 9009, i32 10010, i32 11011, i32 12012, i32 13013, i32 14014, i32 15015, i32 1
6016, i32 17017, i32 18018, i32 19019, i32 20020, i32 21021, i32 22022, i32 23023, i32 24024, i32 25025, i
32 26026, i32 27027, i32 28028, i32 29029, i32 30030, i32 31031, i32 32032, i32 33033, i32 34034, i32 3503
5, i32 36036, i32 37037, i32 38038, i32 39039, i32 40040, i32 41041, i32 42042, i32 43043, i32 44044, i32 
45045, i32 46046, i32 47047, i32 48048, i32 49049, i32 50050, i32 51051, i32 52052, i32 53053, i32 54054, 
...
```

Compiling the sample with a 50MB input file takes about .5 seconds. This is close to the minimum time to compile a file. Adding some binary data doesn't affect build times much, nor should it. A typical translation unit that heavily uses STL may open and load 200,000 lines of code over hundreds of headers. I've measured some range-v3 builds spending 80% of their time in the LLVM optimizer, presumably inlining thousands of function templates away and getting crushed under the bulk of the long mangled names. It's this stuff that slows down builds, not injecting some binary.

## Back to @array

[**embed2.cxx**](embed2.cxx)
```cpp
template<typename type_t>
std::vector<type_t> read_file(const char* filename) {
  FILE* f = fopen(filename, "r");
  if(!f) {
    throw std::runtime_error(format("could not open the file %s:%s", filename));
  }

  // Should really use stat here.
  fseek(f, 0, SEEK_END);
  size_t length = ftell(f);
  fseek(f, 0, SEEK_SET);

  // 128 MB sounds like a lot.
  const size_t max_length = 128<< 20;
  if(length > max_length) {
    throw std::runtime_error(
      format("file %s has length %zu; max allowed is %zu", filename, length,
        max_length));
  }

  // File size must be divisible by type size.
  if(length % sizeof(type_t)) {
    throw std::runtime_error(
      format("file %s has length %zu which is not divisible by size of %s (%zu)",
        filename, length, @type_string(type_t), sizeof(type_t))
    );
  }

  // Read the file in.
  size_t count = length / sizeof(type_t);

  // Size the vector once.
  std::vector<type_t> storage(count);

  // Read the data.
  fread(storage.data(), sizeof(type_t), count, f);

  // Close the file.
  fclose(f);

  // Return the file.
  return std::move(storage);
}

// Read the file.
@meta puts("Reading the file from disk");
@meta auto file_data = read_file<int>("test_binary.data");

// Inject into an array.
@meta puts("Injecting into an array with @array");
const int data[] = @array(file_data.data(), file_data.size());

int main() {
  for(int x : data)
    printf("%d\n", x);

  return 0;
}
```

We can do this without `@embed`. Write an _inline_ function (so that it's not included in the output assembly if only used from a meta context; template functions are implicitly inline) that uses `fopen` and `fread` to read the contents of a file into a vector. Then return that vector and feed it to `@array`. The `@array` keyword now uses the same binary array code paths as `@embed`, so performance should be almost as good, right?

```
$ time circle embed2.cxx
Reading the file from disk
Injecting into an array with @array

real  0m4.160s
user  0m4.012s
sys 0m0.148s
```

Well, no. If you subtract out the `sys` part of the timing (which is mostly the compiler waiting on the OS to open and read files), and just look at the `user` part, the `@array`-oriented code is 12x slower. So what's the deal?

## malloc

[**embed3.cxx**](embed3.cxx)
```cpp
template<typename type_t>
struct malloc_vec_t {
  type_t* data;
  size_t size;
};

template<typename type_t>
malloc_vec_t<type_t> read_file(const char* filename) {
  FILE* f = fopen(filename, "r");
  if(!f) {
    throw std::runtime_error(format("could not open the file %s:%s", filename));
  }

  // Should really use stat here.
  fseek(f, 0, SEEK_END);
  size_t length = ftell(f);
  fseek(f, 0, SEEK_SET);

  // 128 MB sounds like a lot.
  const size_t max_length = 128<< 20;
  if(length > max_length) {
    throw std::runtime_error(
      format("file %s has length %zu; max allowed is %zu", filename, length,
        max_length));
  }

  // File size must be divisible by type size.
  if(length % sizeof(type_t)) {
    throw std::runtime_error(
      format("file %s has length %zu which is not divisible by size of %s (%zu)",
        filename, length, @type_string(type_t), sizeof(type_t))
    );
  }

  // Read the file in.
  size_t count = length / sizeof(type_t);

  // Do a malloc!
  type_t* data = (type_t*)malloc(length);

  // Read the data.
  fread(data, sizeof(type_t), count, f);

  // Close the file.
  fclose(f);

  // Return the file.
  return { data, count };
}
```

I was suspicious of using `std::vector`. It has a constructor that default-initializes its contents, and there's no way to turn that off. What if we replaced that with `malloc`?

```
$ time circle embed3.cxx
Reading the file from disk
Injecting into an array with @array

real  0m0.542s
user  0m0.394s
sys 0m0.147s
```

Now it's only slightly slower than using `@embed`! [embed2.cxx](embed2.cxx) was entirely limited by clearing an array of ints to 0. 

Is `std::vector` just particularly bad? What if we `malloc` the memory and use a _for-statement_ to clear the data to 0:

```
$ time circle embed4.cxx
Reading the file from disk
Injecting into an array with @array

real  0m4.774s
user  0m4.623s
sys 0m0.151s
```

It's actually a bit slower than `std::vector`'s constructor! If we initialized an array of `char`s rather than `int`s it would be again 4x slower.

`memset` would run at full speed, so that won't slow you down. Would `std::fill` run fast or slow? I don't know. One thing specific features like `@embed` do are reduce uncertainty. I **know** `@embed` will run fast, because it only has fast code paths.

## This is not a preprocessor trick

[**embed5.cxx**](embed5.cxx)
```cpp
#include <cstdio>

// Load the file from a string non-type template parameter.
template<const char filename[]>
void do_shader() {
  static const char text[] = @embed(char, filename);
  printf("do_shader<%s> -> %zu bytes\n", filename, sizeof(text));
}

int main() {
  do_shader<"embed1.cxx">();
  do_shader<"embed2.cxx">();
  do_shader<"embed3.cxx">();
  do_shader<"embed4.cxx">();
  do_shader<"embed5.cxx">();
  return 0;
}
```
```
$ circle embed5.cxx
$ ./embed5
do_shader<embed1.cxx> -> 263 bytes
do_shader<embed2.cxx> -> 1911 bytes
do_shader<embed3.cxx> -> 2009 bytes
do_shader<embed4.cxx> -> 2087 bytes
do_shader<embed5.cxx> -> 423 bytes
```

Circle doesn't feature any preprocessor functionality beyond what C++ requires. Extensions like `@embed` operate during definition or instantiation (when an argument is dependent). This is really useful for programmatically injecting code or data into your translation unit.

This sample uses string non-type template parameters to feed `@embed`. The file is loaded only once for each specialization and stored in a static object. You could decay this function to a pointer and stick it in a map to get dynamic lookup of these specializations from a runtime filename.

## What to do

I didn't try to make the interpreter fast, and I was successful in not making it fast. How much effort should be put into that? Probably as much as can be spared. But as more steps are taken to transform the AST into some optimizable graph, you depend more and more on heuristics to indicate if it's a worth-while optimization. These could certainly lead the interpreter astray. Additionally, this is C++ we're talking about optimizing, not a language with a narrow feature set like JavaScript. I don't think we should hope for miracles of performance.

For most stuff, a slow interpreter is fast. You may parse a configuration file of a few-dozen lines and you won't notice the throughput. But on collections of tens of millions of elements, you'll feel the pain.

Circle allows foreign function calls. If you have code that's slow to run, compile it into a shared object and load that at compile time with the command-line argument `-M`. If a function or object is unresolved when used by the interpreter, it will be loaded from the shared object with `dlsym` and executed at full speed. This is a great way to access the full power of the host environment. libc and libstdc++ are pre-loaded, so all your C, C++ and system functions are already ready to be called.

It makes sense to augment the compiler with a _library of features_, that are implemented by the compiler, are generic, and are exposed as keywords. Circle already has more than 50 new keywords, all prefixed by `@`, which establishes a Circle keyword namespace. This can be populated with a lot of great services, and some order can be imposed on the library by using dotted-keywords, eg `@embed.file`, `@embed.folder` and so on. 

The design philosophy of Circle was to provide compile-time control flow, driven by data, allowing you to access information maintained by the compiler. Introspection works this way: use control flow to loop over members of a type, and at each step query the compiler for information about that member. It's only natural that a compiler with rich compile-time control flow would expose much more functionality from the compiler (as distinct from the core language itself), simply because the user could do more with it. Exposing more useful functionality with a compiler-provided library would really improve the programmer experience.

As far as `@embed`, I'll clean some of the internals up (better deduplication, for example), and think about it for a while. Extending it to support a virtual file system, where you could point it at a folder, have it vacuum up the contents of that folder, and expose access to the contents through a literal would be useful. Additionally, patterns for pasing text files, CSV, XML, JSON, YAML and the other usual suspects into appropriate data structures that are accessible at both compile-time and runtime would be super useful. I've long been parsing JSON at compile using [JSON for Modern C++](https://github.com/nlohmann/json]), but the integration and performance leave room for improvement.

Maybe Circle needs a runtime library. .NET, Java, Python and most producitivity languages ship with an extensive library. C++ ships with hardly anything, requiring the user to fend for themselves. A runtime library that was usable at the compiler (via foreign function calls, or, failing that, compiler extensions) would aid program expressiveness and help achieve some performance symmetry.

