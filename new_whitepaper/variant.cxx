#include <iostream>

template<typename type_list>
struct variant_t {
  static_assert(__is_typed_enum(type_list));

  // Create an instance of the enum.
  type_list kind { };

  union {
    // Create a variant member for each enumerator in the type list.
    @meta for enum(auto e : type_list)
      @enum_type(e) @(@enum_name(e));
  };

  // For a real variant, implement the default and copy/move ctors, assignment
  // operators, etc. These use similar for-enum loops to perform actions on the
  // active variant member.

  // Implement a visitor. This calls the callback function and passes the
  // active variant member. f needs to be a generic lambda or function object
  // with a templated operator().
  template<typename func_t>
  auto visit(func_t f) {
    switch(kind) {
      @meta for enum(auto e : type_list) {
        case e:
          return f(@(@enum_name(e)));
          break;      
      }
    }
  }
};

// Define a type list to be used with the variant. Each enumerator identifier
// maps to a variant member name. Each associated type maps to a variant
// member type.
enum typename class my_types_t {
  i = int, 
  d = double,
  s = const char* 
};

int main() {
  auto print_arg = [](auto x) {
    std::cout<< @type_name(decltype(x))<< " : "<< x<< "\n";
  };

  variant_t<my_types_t> var;

  // Fill with a double. var.d is the double variant member.
  var.kind = my_types_t::d;
  var.d = 3.14;
  var.visit(print_arg);

  // Fill with a string. var.s is the const char* variant member.
  var.kind = my_types_t::s;
  var.s = "Hello variant";
  var.visit(print_arg);

  return 0;
}


