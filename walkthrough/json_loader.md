# Walkthrough 3: Deserializing JSON to classes

Consider initializing a program's state from a JSON file of options. On the C++ side, you keep the state in some data structures: aggregates like classes, maps, vectors and sets, which are comprised of other aggregates or scalars like ints, doubles, strings and enums. 

You'd likely organize JSON objects to match their C++ counterparts. Fields in a JSON object have names matching the members in a C++ class, or maybe the keys in an std::map. An array in the JSON corresponds to an array, std::vector, std::set or std::multiset in C++ land. Enums in the JSON are represented by strings that match the spelling of one of the enumerators for the enumeration in C++. 

If we adhere to the set of correspondences, it's very easy to generate JSON->C++ deserialization code automatically using Circle's metaprogramming features. The layout of the C++ classes serves as the schema for deserialization. As we change our internal classes, the schema--and the deserialization code--changes with it.

Of course, you can substitute "JSON" in this walkthrough with your favorite serialization format, like [BSON](http://bsonspec.org/), [YAML](https://yaml.org/) or [Protobuf](https://developers.google.com/protocol-buffers/).

Let's list everything we'd like to support:

* enumerations - in JSON, the value must be a string with a spelling corresponding to the name of one of the enumerators.
* bools - in JSON, the admissable values are `true` and `false`.
* arithmetic types - in JSON, these are all doubles
* std::string
* C arrays and std::array - Fixed-length arrays in C++. The JSON array must be the same lengeth.
* std::vector, std::set, std::multiset - Variable-length arrays in C++. The JSON array may have any number of elements.
* std::map - in JSON, an object of key : value pairs. The key type of the std::map must be std::string, because JSON only permits string-type keys.
* any other class type - in JSON, and object of key : value pairs. Each class member name must have a corresponding key in the JSON object.

Let's throw in another wrinkle--if the C++ type is std::optional, it needn't be represented in the JSON, or the representation may be 'null'. Additionally, if the entity is a class data member with a default initializer, the member needn't be presented in the JSON.

Here's a very simple example of our schema:

[**json_loader.cxx**](json_loader.cxx)
```cpp
enum class language_t {
  english,
  french,
  spanish,
  italian,
  german,
  japanese,
  chinese,
  korean,
};

enum class unit_t {
  mile,
  km,
  league,
  lightyear,
};

struct options_t {
  language_t language;
  unit_t unit = unit_t::km;     // An optional setting.
  std::map<std::string, double> constants;
};
```

The `language` field is a mandatory enum. The `unit` field is an optional enum. The `constants` field is a mandatory map of string/double pairs. A candidate JSON satisfying this schema:

```json
{
  "constants" : {
    "G" : 6.67430e-11,
    "c" : 2.99792488e8,
    "h" : 6.62607015e-34,
    "e" : 1.60217662e-19
  },
  "language" : "spanish"
}
```

## A generic deserializing function

The deserializer is very much the complement of the serializer function `stream_simple` in [eprintf2.cxx](eprintf2.cxx) from the [Walkthrough 2: Evaluating expressions from text](eprintf.md) example. The function recurses over the schema and has special logic for each of the entity types listed above. 

[**util.hxx**](util.hxx)
```cpp
template<typename type_t>
std::optional<type_t> enum_from_name(const char* name) {
  @meta for enum(type_t e : type_t) {
    if(!strcmp(@enum_name(e), name))
      return e;
  }

  return { };
}
```

For example, the handler for enumerations (when `std::is_enum<type_t>::value` is true) calls the `enum_from_name` utility, which string-compares a candidate name to the names of all enumerators. If no match is found, an unset optional is returned.

[**json_loader.cxx**](json_loader.cxx)
```cpp
template<typename type_t>
void load_from_json(nlohmann::json& j, type_t& obj) {

  if constexpr(std::is_enum<type_t>::value) {
    // Match enumerator names.
    if(!j.is_string())
      throw std::runtime_error("expected enum name");

    if(auto e = enum_from_name<type_t>(j.get<std::string>().c_str())) {
      obj = *e;

    } else {
      throw std::runtime_error(format("'%s' is not an enumerator of '%s'\n", 
        j.get<std::string>().c_str(), @type_name(type_t)).c_str());
    }

  } 
```

How does this call fit in with the JSON loader? We first check that the JSON type is a string (we choosen not to support arithmetic enum specifiers). If it's not a string, throw an exception. Call `enum_from_name`, and if that it doesn't find a match, throw a more specific exception indicating the failure. 

```cpp
  else {
    static_assert(std::is_class<type_t>::value, "expected a class object");

    if(!j.is_object())
      throw std::runtime_error("expected object");

    // Initialize using each member from the class.
    @meta for(int i = 0; i < @member_count(type_t); ++i) {
      // Lookup the key for each member name.
      auto it = j.find(@member_name(type_t, i));
      if(j.end() != it) {
        // There is a key in the JSON file. Use it to initialize the member.
        try { 
          load_from_json(*it, @member_ref(obj, i));

        } catch(std::runtime_error e) {
          throw std::runtime_error(format(".%s: %s", @member_name(type_t, i),
            e.what()).c_str());
        }

      } else if constexpr(
        !@is_class_template(@member_type(type_t, i), std::optional) &&
        !@member_has_default(type_t, i)) {

        // Error if the member is not std::optional and it doesn't have
        // a default initializer.
        throw std::runtime_error(
          format("missing key '%s'\n", @member_name(type_t, i)).c_str());
      }
    }
  }
```

The code that splits apart classes drives the deserializer. For a generic class type (i.e. not one of the special cases we carved out), we use a compile-time loop to visit each non-static data member. We search the JSON object for a key matching the member name. If one exists, we recursive invoke `load_from_json`, providing both the JSON of the value and a reference to the object's data member.

If there's no key corresponding to our data member, we'll check if it's an _optional_ member. If the member is an instance of std::optional we won't throw an error. Likewise, if the member has a default initializer, as told by the new compiler intrinsic `@member_has_default`, we'll continue deserialization. (NB: `member_default` yields the expression of the default initializer.)

It's important to provide the user with helpful errors, especially when the schema is elaborate or may change with time. To do this, we throw exceptions from scalar handlers, and catch the errors, append additional context in the error string, and rethrow in the aggregate handlers.

```cpp
    else if constexpr(std::is_array<type_t>::value || 
    @is_class_template(type_t, std::array)) {

    if(!j.is_array())
      throw std::runtime_error("expected array");

    // Handle arrays and std::array.
    // Gonfirm that the size matches the extent.
    const size_t extent = std::extent<type_t, 0>::value;
    size_t size = j.size();

    if(size != extent) {
      throw std::runtime_error(
        format("array of size %zu where %zu expected", size, extent).c_str());
    }

    // Load in each arary element.
    for(size_t i = 0; i < extent; ++i) {
      try {
        load_from_json(j[i], obj[i]); 

      } catch(std::runtime_error e) {
        throw std::runtime_error(format("[%zu]: %s", i, e.what()));
      }
    }
```
Read through the code for handling arrays and std::array. We throw errors if the JSON is not an array, or is an array of unexpected size. When both of those checks pass, we enter our loop and recursively use `load_from_json` to deserialize each array element. If any of these operations fail, the exception is caught in a handler defined for that one element, and new exception is thrown holding an error specifying both the value of the array subscript and the text of the caught exception.

```json
{
  "constants" : {
    "G" : 6.67430e-11,
    "c" : 2.99792488e8,
    "h" : 6.62607015e-34,
    "e" : 1.60217662e-19
  },
  "language" : "spanesh"
}
```
```
$ ./json_loader 
.language: 'spanesh' is not an enumerator of 'language_t'
```

If the user botches the spelling of "spanish" in his configuration file, they'll receive a useful diagonstic of what went wrong: not just the failed spelling and the enumeration name, but also the data member of the value.

```cpp
  if constexpr(std::is_enum<type_t>::value) {
    // Match enumerator names.
    if(!j.is_string())
      throw std::runtime_error("expected enum name");

    if(auto e = enum_from_name<type_t>(j.get<std::string>().c_str())) {
      obj = *e;

    } else {
      std::ostringstream oss;
      oss<< '\''<< j.get<std::string>() << "\' is not an enumerator of '"<< 
        @type_name(type_t)<< "\n";
      oss<< "Did you mean\n";
      oss<< "  "<< @enum_names(type_t)<< "\n" ...;
      oss<< "?\n";

      throw std::runtime_error(oss.str());
    }
  } 
```
```
$ ./json_loader 
.language: 'spanesh' is not an enumerator of 'language_t
Did you mean
  english
  french
  spanish
  italian
  german
  japanese
  chinese
  korean
?
```
If we're feeling chatty we can list all the enumerator names as a diagnostic. This is deftly accomplished using the `@enum_names` intrinsic, which returns string literals for the names of each enumerator as a parameter pack. We can expand that pack in an expression statement to effect a loop that prints all the enumerator names as suggestions.

The following listing is a completely implements JSON deserialization for all of the special cases listed at the top of the document. It shouldn't be difficult adding additional handlers for your own cases of interest, such as matrices, or file paths or mathematical functions. The value of this library is that it evolves with your code--just change the internal C++ objects, recompile, and voila, the deserialization is up to date. Run your old JSON files through and you'll get helpful diagnostics on how to migrate to the new schema.

[**json_loader.cxx**](json_loader.cxx)
```cpp
#include "util.hxx"
#include <fstream>
#include <iostream>
#include <array>
#include <map>
#include <set>
#include <optional>
#include "json.hpp"

// Return an object by reference. We do this because we can't return an 
// array type, and arrays are otherwise supported.

template<typename type_t>
void load_from_json(nlohmann::json& j, type_t& obj) {

  if constexpr(std::is_enum<type_t>::value) {
    // Match enumerator names.
    if(!j.is_string())
      throw std::runtime_error("expected enum name");

    if(auto e = enum_from_name<type_t>(j.get<std::string>().c_str())) {
      obj = *e;

    } else {
      /*
      throw std::runtime_error(format("'%s' is not an enumerator of '%s'\n", 
        j.get<std::string>().c_str(), @type_name(type_t)).c_str());
      */

      std::ostringstream oss;
      oss<< '\''<< j.get<std::string>() << "\' is not an enumerator of '"<< 
        @type_name(type_t)<< "\n";
      oss<< "Did you mean\n";
      oss<< "  "<< @enum_names(type_t)<< "\n" ...;
      oss<< "?\n";

      throw std::runtime_error(oss.str());
    }

  } else if constexpr(std::is_same<bool, type_t>::value) {
    if(!j.is_boolean())
      throw std::runtime_error("expected boolean value");

    obj = j.get<bool>();

  } else if constexpr(std::is_arithmetic<type_t>::value) {
    if(!j.is_number())
      throw std::runtime_error("expected numeric value");

    obj = j.get<type_t>();

  } else if constexpr(std::is_same<std::string, type_t>::value) {
    if(!j.is_string())
      throw std::runtime_error("expected numeric value");

    obj = j.get<std::string>();

  } else if constexpr(std::is_array<type_t>::value || 
    @is_class_template(type_t, std::array)) {

    if(!j.is_array())
      throw std::runtime_error("expected array");

    // Handle arrays and std::array.
    // Gonfirm that the size matches the extent.
    const size_t extent = std::extent<type_t, 0>::value;
    size_t size = j.size();

    if(size != extent) {
      throw std::runtime_error(
        format("array of size %zu where %zu expected", size, extent).c_str());
    }

    // Load in each arary element.
    for(size_t i = 0; i < extent; ++i) {
      try {
        load_from_json(j[i], obj[i]); 

      } catch(std::runtime_error e) {
        throw std::runtime_error(format("[%zu]: %s", i, e.what()));
      }
    }
    
  } else if constexpr(@is_class_template(type_t, std::vector)) {
    if(!j.is_array())
      throw std::runtime_error("expected array");

    // Resize the object's vector and call load_from_json on each element.   
    size_t size = j.size();
    obj.resize(size);
    for(size_t i = 0; i < size; ++i) {
      try {
        load_from_json(j[i], obj[i]);

      } catch (std::runtime_error e) {
        throw std::runtime_error(format("[%zu]: %s", i, e.what()));
      }
    }

  } else if constexpr(@is_class_template(type_t, std::optional)) {
    // Load the inner type.
    if(!j.is_null())
      load_from_json(j, *obj);

  } else if constexpr(@is_class_template(type_t, std::set) ||
    @is_class_template(type_t, std::multiset)) {
    // Like an array, but unordered.
    if(!j.is_array())
      throw std::runtime_error("expected array");

    size_t size = j.size();
    for(size_t i = 0; i < size; ++i) {
      typename type_t::value_type value { };

      try {
        load_from_json(j[i], value);

      } catch (std::runtime_error e) {
        throw std::runtime_error(format("[%zu]: %s", i, e.what()));
      }

      obj.insert(std::move(value));
    }

  } else if constexpr(@is_class_template(type_t, std::map)) {
    // Read out all the key-value pairs.
    typedef typename type_t::key_type key_type;
    typedef typename type_t::mapped_type value_type;

    static_assert(std::is_same<key_type, std::string>::value,
      "key type of std::map must be std::string to support json serialization");

    if(!j.is_object())
      throw std::runtime_error("expected object");

    for(auto pair : j.items()) {
      key_type key = pair.key();
      value_type value { };

      try {
        load_from_json(pair.value(), value);

      } catch(std::runtime_error e) {
        throw std::runtime_error(format("[%s]: %s", key.c_str(), e.what()));

      }

      // Attempt to insert into the map.
      obj.insert(std::make_pair(std::move(key), std::move(value)));
    }   

  } else {
    static_assert(std::is_class<type_t>::value, "expected a class object");

    if(!j.is_object())
      throw std::runtime_error("expected object");

    // Initialize using each member from the class.
    @meta for(int i = 0; i < @member_count(type_t); ++i) {
      // Lookup the key for each member name.
      auto it = j.find(@member_name(type_t, i));
      if(j.end() != it) {
        // There is a key in the JSON file. Use it to initialize the member.
        try { 
          load_from_json(*it, @member_ref(obj, i));

        } catch(std::runtime_error e) {
          throw std::runtime_error(format(".%s: %s", @member_name(type_t, i),
            e.what()).c_str());
        }

      } else if constexpr(
        !@is_class_template(@member_type(type_t, i), std::optional) &&
        !@member_has_default(type_t, i)) {

        // Error if the member is not std::optional and it doesn't have
        // a default initializer.
        throw std::runtime_error(
          format("missing key '%s'\n", @member_name(type_t, i)).c_str());
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

enum class language_t {
  english,
  french,
  spanish,
  italian,
  german,
  japanese,
  chinese,
  korean,
};

enum class unit_t {
  mile,
  km,
  league,
  lightyear,
};

struct options_t {
  language_t language;
  unit_t unit = unit_t::km;     // An optional setting.
  std::map<std::string, double> constants;
};

int main() {

  try {

    std::ifstream f("options.json");
    nlohmann::json j;
    f>> j;

    options_t options { };
    load_from_json(j, options);

    std::cout<< "{options}"_e<< "\n";

  } catch(std::exception& e) {
    std::cout<< e.what()<< "\n";
  }
  return 0;
}
```

