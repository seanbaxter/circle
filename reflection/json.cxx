#include "json.hpp"

using namespace nlohmann;

typedef float    vec2    __attribute__((vector_size(8)));
typedef float    vec3    __attribute__((vector_size(12)));
typedef float    vec4    __attribute__((vector_size(16)));

template<typename type_t>
type_t load_from_json(json& j) {
  if constexpr(std::is_class_v<type_t>) {

  } else if constexpr(__is_vector(type_t)) {
    static_assert(std::is_same_v<float, __underlying_type(type_t)>);
    constexpr int size = __vector_size(type_t);

    check()
  }
}

template<typename type_t>
inline void check(bool valid, const std::string& name, 
  const std::string error) {

  if(!valid) {
    fprintf(stderr, "%s: %s\n", name.c_str(), error.c_str());
    exit(1);
  }
}

// Load a class object consisting of class objects, vectors and scalars from
// a JSON.
template<typename obj_t>
obj_t load_from_json(std::string name, nlohmann::json& j) {
  obj_t obj { };

  if(j.is_null()) {
    fprintf(stderr, "no JSON item for %s\n", name.c_str());
    exit(1);
  }

  if constexpr(std::is_class_v<obj_t>) {
    // Read any class type.
    check(j.is_object(), name, "expected object type");
    @meta for(int i = 0; i < @member_count(obj_t); ++i)
      obj.@member_value(i) = load_from_json<@member_type(obj_t, i)>(
        name + "." + @member_name(obj_t, i),
        j[@member_name(obj_t, i)]
      );

  } else if constexpr(__is_vector(obj_t)) {
    constexpr int size = __vector_size(obj_t);

    check(j.is_array(), name, "expected array type");
    check(j.size() == size, name, 
      "expected " + std::to_string(size) + " array elements");

    for(int i = 0; i < size; ++i) {
      obj[i] = load_from_json<__underlying_type(obj_t)>(
        name + "[" + std::to_string(i) + "]",
        j[i]
      );
    }

  } else {
    static_assert(std::is_integral_v<obj_t> || std::is_floating_point_v<obj_t>);
    check(j.is_number(), name, "expected number type");
    obj = j;
  }
  return obj;
} 

// Use + to get a count of the occurrences for a type in the list.
template<typename type_t, typename list_t>
constexpr size_t occurence_in_list_v =
  (... + (size_t)std::is_same_v<type_t, @enum_types(list_t)>);

// True if the list has no duplicate types.
template<typename list_t>
constexpr bool is_unique_list = 
  (... && (1 == occurence_in_list_v<@enum_types(list_t), list_t>));

template<typename typelist_t>
auto load_variant_from_json(j) {
  static_assert(__is_typed_enum(typelist_t), 
    "load_from_json argument must be a typed enum");
  static_assert(is_unique_list<typelist_t>,
    "load_from_json typed enum argument must have unique associated types");

  std::variant<@enum_types(typelist_t)...> var;

  std::string type = j["type"].str();
  @meta for enum(my_types_t e : my_types_t) {
    if(@enum_type_string(e) == )
  }
}