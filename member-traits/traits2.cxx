#include <iostream>
#include <limits>

#define PRINT_TRAIT(type, limit) \
  std::cout<< type.string + "." #limit + " = "<< type.limit<< "\n";

template<typename type_t>
void print_type_traits() {
  // From std::numeric_limits
  if constexpr(type_t.has_numeric_limits) {
    PRINT_TRAIT(type_t, is_signed)
    PRINT_TRAIT(type_t, is_integer)
    PRINT_TRAIT(type_t, is_exact)
    PRINT_TRAIT(type_t, has_infinity)
    PRINT_TRAIT(type_t, has_quiet_NaN)
    PRINT_TRAIT(type_t, has_signaling_NaN)
    PRINT_TRAIT(type_t, has_denorm)
    PRINT_TRAIT(type_t, has_denorm_loss)
    PRINT_TRAIT(type_t, round_style)
    PRINT_TRAIT(type_t, is_iec559)
    PRINT_TRAIT(type_t, is_bounded)
    PRINT_TRAIT(type_t, is_modulo)
    PRINT_TRAIT(type_t, digits)
    PRINT_TRAIT(type_t, digits10)
    PRINT_TRAIT(type_t, max_digits10)
    PRINT_TRAIT(type_t, radix)
    PRINT_TRAIT(type_t, min_exponent)
    PRINT_TRAIT(type_t, min_exponent10)
    PRINT_TRAIT(type_t, max_exponent)
    PRINT_TRAIT(type_t, max_exponent10)
    PRINT_TRAIT(type_t, traps)
    PRINT_TRAIT(type_t, tinyness_before)
    PRINT_TRAIT(type_t, min)
    PRINT_TRAIT(type_t, lowest)
    PRINT_TRAIT(type_t, max)
    PRINT_TRAIT(type_t, epsilon)
    PRINT_TRAIT(type_t, round_error)
    PRINT_TRAIT(type_t, infinity)
    PRINT_TRAIT(type_t, quiet_NaN)
    PRINT_TRAIT(type_t, signaling_NaN)
    PRINT_TRAIT(type_t, denorm_min)
  }

  // From <type_traits>
  PRINT_TRAIT(type_t, is_void)
  PRINT_TRAIT(type_t, is_null_pointer)
  PRINT_TRAIT(type_t, is_integral)
  PRINT_TRAIT(type_t, is_floating_point)
  PRINT_TRAIT(type_t, is_array)
  PRINT_TRAIT(type_t, is_enum)
  PRINT_TRAIT(type_t, is_union)
  PRINT_TRAIT(type_t, is_class)
  PRINT_TRAIT(type_t, is_function)
  PRINT_TRAIT(type_t, is_pointer)
  PRINT_TRAIT(type_t, is_lvalue_reference)
  PRINT_TRAIT(type_t, is_rvalue_reference)
  PRINT_TRAIT(type_t, is_member_object_pointer)
  PRINT_TRAIT(type_t, is_member_function_pointer)
  PRINT_TRAIT(type_t, is_fundamental)
  PRINT_TRAIT(type_t, is_arithmetic)
  PRINT_TRAIT(type_t, is_scalar)
  PRINT_TRAIT(type_t, is_object)
  PRINT_TRAIT(type_t, is_compound)
  PRINT_TRAIT(type_t, is_reference)
  PRINT_TRAIT(type_t, is_member_pointer)
  PRINT_TRAIT(type_t, is_const)
  PRINT_TRAIT(type_t, is_volatile)
  PRINT_TRAIT(type_t, is_trivial)
  PRINT_TRAIT(type_t, is_trivially_copyable)
  PRINT_TRAIT(type_t, is_standard_layout)
  PRINT_TRAIT(type_t, is_empty)
  PRINT_TRAIT(type_t, is_polymorphic)
  PRINT_TRAIT(type_t, is_abstract)
  PRINT_TRAIT(type_t, is_final)
  PRINT_TRAIT(type_t, is_aggregate)
  PRINT_TRAIT(type_t, is_unsigned)
  PRINT_TRAIT(type_t, is_bounded_array)
  PRINT_TRAIT(type_t, is_unbounded_array)
  PRINT_TRAIT(type_t, rank)
}

#undef PRINT_LIMIT


int main() {
  // Print a type with numeric_limits.
  print_type_traits<double>();

  // Print a type without numeric limits.
  print_type_traits<char*>();
}
