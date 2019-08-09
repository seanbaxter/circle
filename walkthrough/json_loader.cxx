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
