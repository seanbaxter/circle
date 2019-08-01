#include <algorithm>
#include <vector>
#include <cstdio>

template<typename... types_t>
struct foo_t {
  @meta printf("%s\n", @type_name(types_t)) ...;
};

int main() {

  @meta const char* type_names[] {
    "int",
    "double",
    "void*",
    "double",
    "char[3]",
    "int",
    "const char[5]",
    "char[3]"
  };

  // Convert strings to mtypes.
  @meta std::vector<@mtype> types {
    @dynamic_type(@type_id(@pack_nontype(type_names))) ...
  };

  // Sort and unique the types.
  @meta std::sort(types.begin(), types.end());
  @meta types.erase(std::unique(types.begin(), types.end()), types.end());

  // Instantiate foo_t on the unique types.
  foo_t<@pack_type(types)...> foo;

  return 0;
}