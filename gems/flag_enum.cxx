#include <cstdio>
#include <cstdint>

// Generate an enum with flag-like value for each string.
// Infer the number of strings in the array by deducing a template
// argument.
template<size_t count>
@macro void define_flags(const char* (&flag_names)[count]) {
  // Loop over each string in the array.
  @meta for(size_t i = 0; i < count; ++i)
    // Declare an enumerator and give it a flag-like value.
    @(flag_names[i]) = 1<< i;
}

// This macro defines an enumeration and overloaded operator| and & for that
// type into the namespace in which it's expanded.
template<typename underlying_t = uint32_t, size_t count = 0>
@macro void define_flag_enum(const char* name, 
  const char* (&flag_names)[count]) {

  // Define the scoped enum.
  enum class @(name) : underlying_t {
    // Declare each flag-like enumerator.
    @macro define_flags(my_flag_names);
  };

  // Overload operator|
  @(name) operator|(@(name) lhs, @(name) rhs) {
    return static_cast<@(name)>((underlying_t)lhs | (underlying_t)rhs);
  }

  // Overload operator&
  @(name) operator&(@(name) lhs, @(name) rhs) {
    return static_cast<@(name)>((underlying_t)lhs & (underlying_t)rhs);
  }
}

// Generate enums with these flag names. 
@meta const char* my_flag_names[] {
  "a", "b", "c", "d", "e"
};

// Define my_enum_t in global ns. The underlying type defaults to uint32_t.
@macro define_flag_enum("my_enum_t", my_flag_names);

namespace ns {
  // Define my_enum2_t in namespace ns. Supply a uint16_t underlying type.
  @macro define_flag_enum<uint16_t>("my_enum2_t", my_flag_names);
}

int main() {
  // Invoke operator| for ns::my_enum2_t. The overloaded operator| is defined
  // in namespace ns, but is found through argument-dependent lookup.
  ns::my_enum2_t x = ns::my_enum2_t::c | ns::my_enum2_t::d;

  return 0;
}
