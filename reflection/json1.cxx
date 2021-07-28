#include "json.hpp"
#include "print.hxx"
#include <fstream>

using namespace nlohmann;

using alt_name [[attribute]] = const char*;

template<typename enum_t>
std::optional<enum_t> string_to_enum(const char* s) {
  @meta for enum(enum_t e : enum_t) {

    if constexpr(@enum_has_attribute(e, alt_name)) {
      if(0 == strcmp(@enum_attribute(e, alt_name), s))
        return e;
    }

    if(0 == strcmp(e.string, s))
      return e;
  }
  return { };
}

#define ERROR(pattern, ...) \
  { fprintf(stderr, pattern, __VA_ARGS__); exit(1); }


template<typename type_t>
type_t load_from_json(json& j) {
  type_t obj { };

  if constexpr(std::string == type_t) {
    obj = j.get<std::string>();

  } else if constexpr(std::is_class_v<type_t>) {

    // Don't support classes with non-public members.
    static_assert(
      !@member_count(type_t, protected private),
      "cannot stream type \"" + type_t.string + 
        "\" with non-public member objects"
    );

    // Don't support classes with bases.
    static_assert(
      !@base_count(type_t, all),
      "cannot stream type \""s + type_t.string + "\" with base classes"
    );

    // Loop over data members.
    @meta for(int i = 0; i < @member_count(type_t); ++i) {
      // Look up a value.
      json& j2 = j[@member_name(type_t, i)];

      // Error if it's not available and with no defaulted.
      if(!j2.is_null()) {
        obj.@member_value(i) = load_from_json<@member_type(type_t, i)>(j2);

      } else if constexpr(!@member_has_default(type_t, i))
        ERROR("JSON missing member %s\n", @member_name(type_t, i));
    }

  } else if constexpr(std::is_array_v<type_t>) {
    typedef std::remove_extent_t<type_t> inner_t;
    size_t i = 0;
    for(json& j2 : j) 
      obj[i++] = load_from_json<inner_t>(j2);

  } else if constexpr(std::is_enum_v<type_t>) {
    if(auto e = string_to_enum<type_t>(j.get<std::string>().c_str())) {
      obj = *e;

    } else {
      ERROR("%s is not a %s enumerator\n", j.get<std::string>().c_str(), 
        @type_name(type_t));
    }

  } else if constexpr(bool == type_t) {
    obj = j.get<bool>();

  } else {
    static_assert(std::is_arithmetic_v<type_t>);
    obj = j.get<type_t>();
  }

  return obj;
}

enum class weightclass_t {
  featherweight,
  lightweight,
  welterweight,
  middleweight,
  heavyweight,

  // We can't use a hyphen in a c++ identifier, but we can in a JSON key.
  // Associate this alternate name using an attribute.
  lheavyweight [[.alt_name="light-heavyweight"]],
};

enum class stance_t {
  orthodox,
  southpaw,
};

struct boxer_t {
  std::string name;
  weightclass_t weight;
  int height;
  int reach;

  // By providing a default here, we don't error if the JSON doesn't have one.
  stance_t stance = stance_t::orthodox;
};

int main() {
  std::ifstream file("boxers.json");
  json j;
  file>> j;

  size_t i = 0;
  for(json j2 : j["boxers"]) {
    printf("Object %d:\n", i++);
    boxer_t boxer = load_from_json<boxer_t>(j2);
    print_object(boxer, 1);

  }
}