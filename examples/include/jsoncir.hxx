#pragma once
#include <optional>
#include <functional>
#include "json.hpp"
#include "enums.hxx"
#include "format.hxx"

using nlohmann::json;
typedef std::optional<std::reference_wrapper<json> > json_ref;

////////////////////////////////////////////////////////////////////////////////
// Generic functions for finding JSON items by key and reading out values.
// This should go in a header file--it's generic and runs at compile-time or
// runtime.

// Test that each member of the key is represented in the JSON item and has 
// the same value.
template<typename key_t>
bool compare_key(json& item, const key_t& key) {
  @meta for(int i = 0; i < @member_count(key_t); ++i) {
    if(item.count(@member_name(key_t, i))) {
      auto x = item[@member_name(key_t, i)].template get<@member_type(key_t, i)>();
      if(@member_ref(key, i) != x)
        return false;
    }
  }

  return true;
}

// Search for a top-level item in the global json file j that matches
// the key.
template<typename key_t>
json_ref find_json_item(json& j, const key_t& key) {
  // Iterate through the items in the json file and find the first that has
  // all the requested key/value pairs.
  for(json& j2 : j) {
    if(compare_key(j2, key))
      return j2;
  }

  cirprint("  **No JSON item matching key %\n", key);
  return { };
}

// Read all the fields of the type_t class from the JSON item.
// Handle enums and vector types specially.
template<typename type_t, typename key_t>
std::optional<type_t> read_json_value(json& item, const key_t& key, 
  const char* name) {

  std::optional<type_t> value { };
  if constexpr(std::is_enum<type_t>::value) {
    // The JSON item must be a string that matches the spelling of one
    // of the enumerators.
    std::string s = item.get<std::string>();
    if(auto e = enum_from_name<type_t>(s.c_str()))
      value = *e;
    else
      cirprint("  **Unrecognized enum '%' at '%' in %\n", s, name, key);
    
  } else if constexpr(@is_class_template(type_t, std::vector>)) {
    if(item.is_array()) {
      // Read each member of the JSON array.
      typedef typename type_t::value_type inner_t;
      type_t vec;
      for(auto& e : item) {
        if(auto val = read_json_value<inner_t>(e, key, name))
          vec.push_back(std::move(*val));
      }
      value = std::move(vec);

    } else
      cirprint("  **Expected array at '%' in %\n", name, key);

  } else
    value = item.get<type_t>();

  return value;
}

// Extract the value for each data member in value_t. If one isn't there,
// print an error.
template<typename value_t, typename key_t>
value_t load_json_value(json& item, const key_t& key) {
  value_t value { };
  @meta for(int i = 0; i < @member_count(value_t); ++i) {{
    // Use an extra { to open a real scope to hold the name variable.
    const char* name = @member_name(value_t, i);
    if(item.count(name)) {
      // Extract the value from the JSON if it exists.
      typedef @member_type(value_t, i) type_t;
      if(auto val = read_json_value<type_t>(item.at(name), key, name))
        @member_ref(value, i) = std::move(*val);

    } else
      cirprint("  **No field '%' in %\n", name, key);
  }}

  return value;
}

// Search for a top-level JSON item matching the key. Then extract the
// requested items and return them in a structure.
template<typename value_t, typename key_t>
value_t find_json_value(json& j, const key_t& key) {
  if(auto item = find_json_item(j, key))
    return load_json_value<value_t>(*item, key);
  return { };
}