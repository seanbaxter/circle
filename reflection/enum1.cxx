#include <cstdio>

enum typename class shapes_t {
  int, double, char[5],
};

template<typename enum_t>
void list_enums() {
  @meta for enum(enum_t e : enum_t) {
    printf("%s:\n  type=%s type-string=%s decl-string=%s\n", 
      @enum_name(e),
      @type_string(@enum_type(e)),
      @enum_type_string(e),
      @enum_decl_string(e)
    );
  }  
}

int main() {
  list_enums<shapes_t>();
}