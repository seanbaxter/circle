#include <cstdio>
#include <cstring>
#include <optional>
#include <type_traits>

template<typename type_t>
std::optional<type_t> name_to_enum(const char* name) {
  static_assert(std::is_enum<type_t>::value, 
    "name_to_enum must be called on enum type");

  // Loop over each enumerator in type_t at compile time.
  @meta for(size_t i = 0; i < @enum_count(type_t); ++i) {
    // Compare the provided name to the enumerator's name.
    if(!strcmp(@enum_name(type_t, i), name))
      return @enum_value(type_t, i);
  }

  return { };
}

int main() {
  enum my_enum_t {
    a, b, c, d, e
  };

  const char* name = "d";
  if(auto e = name_to_enum<my_enum_t>(name)) {
    printf("matched enumerator '%s' with value '%d'\n", name, (int)*e);

  } else {
    printf("'%s' is not an enumerator of 'my_enum_t'\n");
  }
 
  return 0;
}
