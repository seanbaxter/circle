#include "json.hpp"
#include <fstream>

using namespace nlohmann;

// Load types.json at compile time. All meta statements are executed at 
// compile time.
@meta std::ifstream file("types.json");
@meta json j;
@meta file>> j;

@meta for(json& types : j["types"]) {

  // Use a dynamic name to turn the JSON "name" value into an identifier.
  struct @(types["name"]) {

    // Loop over the JSON "members" array.
    @meta for(json& members : types["members"]) {
      // Emit each member. The inner-most enclosing non-meta scope is the
      // class-specifier, so this statement is a member-specifier.
      @type_id(members["type"]) @(members["name"]);
    }
  };
}

// Make a typed enum to keep a record of the types we injected.
enum typename new_types_t {
  @meta for(json& types : j["types"])
    @type_id(types["name"]);
};

int main() {
  @meta for enum(new_types_t e : new_types_t) {
    printf("struct %s {\n", @enum_type_string(e));
    printf("  %s;\n", @member_decl_strings(@enum_type(e)))...;
    printf("};\n");
  }
}