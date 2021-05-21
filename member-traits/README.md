# Circle member traits

Member traits is a Circle extension that makes using `numeric_limits` and `type_traits` fun and easy. These constants and types are accessed like non-static data members on a _type_.

[**traits1.cxx**](traits1.cxx)
```cpp
#include <limits>
#include <type_traits>
#include <array>
#include <list>
#include <iostream>

struct base_t { virtual void func() = 0; };
struct derived_t final : base_t { virtual void func() override { } };
enum enum_t : short { };

int main() {
  // Access all numeric_traits.
  std::cout<< float.max        << ", "<< float.epsilon<< "\n";
  std::cout<< double.max       << ", "<< double.epsilon<< "\n";
  std::cout<< (long double).max<< ", "<< (long double).epsilon<< "\n";

  // Access type_traits.
  static_assert(void.is_void);
  static_assert(std::array<float, 5>.is_aggregate);
  static_assert(!std::list<char>.is_aggregate);
  static_assert((float[][2][3]).is_unbounded_array);
  static_assert((long(std::list<char>::*)).is_member_object_pointer);
  static_assert(derived_t.is_final);
  static_assert(base_t.is_polymorphic);
  static_assert(enum_t.is_enum);

  // Mutate types using type_traits.
  static_assert(std::is_same_v<
    const int*.remove_pointer.make_unsigned,
    const unsigned int
  >);

  static_assert(std::is_same_v<
    enum_t.underlying_type.add_rvalue_reference,
    short&&
  >);

  static_assert(std::is_same_v<
    char(.decay)[3],      // This is how to write (char[3]).decay.
    char*
  >);
}
```
```
$ circle -std=c++20 traits1.cxx && ./traits1
3.40282e+38, 1.19209e-07
1.79769e+308, 2.22045e-16
1.18973e+4932, 1.0842e-19
```

There are three kinds of traits supported with this extension:
1. **Value members of `std::numeric_limits`.** These are only supported for types that have a partial or explicit specialization of the `std::numeric_limits` class template. Attempting to accessa numeric limit on a type that doesn't have a specialization (by default, any non-builtin or non-arithmetic type) raises a SFINAE failure. The `has_numeric_limits` trait indicates if the type on the left-hand side has a specialization.
    * `has_numeric_limits` - a special trait that indicates if the numeric limits are available.
    * [`is_signed`](https://en.cppreference.com/w/cpp/types/numeric_limits/is_signed)
    * [`is_integer`](https://en.cppreference.com/w/cpp/types/numeric_limits/is_integer)
    * [`is_exact`](https://en.cppreference.com/w/cpp/types/numeric_limits/is_exact)
    * [`has_infinity`](https://en.cppreference.com/w/cpp/types/numeric_limits/has_infinity)
    * [`has_quiet_NaN`](https://en.cppreference.com/w/cpp/types/numeric_limits/has_quiet_NaN)
    * [`has_signaling_NaN`](https://en.cppreference.com/w/cpp/types/numeric_limits/has_signaling_NaN)
    * [`has_denorm`](https://en.cppreference.com/w/cpp/types/numeric_limits/has_denorm)
    * [`has_denorm_loss`](https://en.cppreference.com/w/cpp/types/numeric_limits/has_denorm_loss)
    * [`round_style`](https://en.cppreference.com/w/cpp/types/numeric_limits/round_style)
    * [`is_iec559`](https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559)
    * [`is_bounded`](https://en.cppreference.com/w/cpp/types/numeric_limits/is_bounded)
    * [`is_modulo`](https://en.cppreference.com/w/cpp/types/numeric_limits/is_modulo)
    * [`digits`](https://en.cppreference.com/w/cpp/types/numeric_limits/digits)
    * [`digits10`](https://en.cppreference.com/w/cpp/types/numeric_limits/digits10)
    * [`max_digits10`](https://en.cppreference.com/w/cpp/types/numeric_limits/max_digits10)
    * [`radix`](https://en.cppreference.com/w/cpp/types/numeric_limits/radix)
    * [`min_exponent`](https://en.cppreference.com/w/cpp/types/numeric_limits/min_exponent)
    * [`min_exponent10`](https://en.cppreference.com/w/cpp/types/numeric_limits/min_exponent10)
    * [`max_exponent`](https://en.cppreference.com/w/cpp/types/numeric_limits/max_exponent)
    * [`max_exponent10`](https://en.cppreference.com/w/cpp/types/numeric_limits/max_exponent10)
    * [`traps`](https://en.cppreference.com/w/cpp/types/numeric_limits/traps)
    * [`tinyness_before`](https://en.cppreference.com/w/cpp/types/numeric_limits/tinyness_before)
    * [`min`](https://en.cppreference.com/w/cpp/types/numeric_limits/min)
    * [`lowest`](https://en.cppreference.com/w/cpp/types/numeric_limits/lowest)
    * [`max`](https://en.cppreference.com/w/cpp/types/numeric_limits/max)
    * [`epsilon`](https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon)
    * [`round_error`](https://en.cppreference.com/w/cpp/types/numeric_limits/round_error)
    * [`infinity`](https://en.cppreference.com/w/cpp/types/numeric_limits/infinity)
    * [`quiet_NaN`](https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN)
    * [`signaling_NaN`](https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN)
    * [`denorm_min`](https://en.cppreference.com/w/cpp/types/numeric_limits/denorm_min)
    
2. **Value members of `std::type_traits`.** These are supported for all C++ types. Some of these traits require C++20 or even C++23 headers. 
    * [`is_void`](https://en.cppreference.com/w/cpp/types/is_void)
    * [`is_null_pointer`](https://en.cppreference.com/w/cpp/types/is_null_pointer)
    * [`is_integral`](https://en.cppreference.com/w/cpp/types/is_integral)
    * [`is_floating_point`](https://en.cppreference.com/w/cpp/types/is_floating_point)
    * [`is_array`](https://en.cppreference.com/w/cpp/types/is_array)
    * [`is_enum`](https://en.cppreference.com/w/cpp/types/is_enum)
    * [`is_union`](https://en.cppreference.com/w/cpp/types/is_union)
    * [`is_class`](https://en.cppreference.com/w/cpp/types/is_class)
    * [`is_function`](https://en.cppreference.com/w/cpp/types/is_function)
    * [`is_pointer`](https://en.cppreference.com/w/cpp/types/is_pointer)
    * [`is_lvalue_reference`](https://en.cppreference.com/w/cpp/types/is_lvalue_reference)
    * [`is_rvalue_reference`](https://en.cppreference.com/w/cpp/types/is_rvalue_reference)
    * [`is_member_object_pointer`](https://en.cppreference.com/w/cpp/types/is_member_object_pointer)
    * [`is_member_function_pointer`](https://en.cppreference.com/w/cpp/types/is_member_function_pointer)
    * [`is_fundamental`](https://en.cppreference.com/w/cpp/types/is_fundamental)
    * [`is_arithmetic`](https://en.cppreference.com/w/cpp/types/is_arithmetic)
    * [`is_scalar`](https://en.cppreference.com/w/cpp/types/is_scalar)
    * [`is_object`](https://en.cppreference.com/w/cpp/types/is_object)
    * [`is_compound`](https://en.cppreference.com/w/cpp/types/is_compound)
    * [`is_reference`](https://en.cppreference.com/w/cpp/types/is_reference)
    * [`is_member_pointer`](https://en.cppreference.com/w/cpp/types/is_member_pointer)
    * [`is_const`](https://en.cppreference.com/w/cpp/types/is_const)
    * [`is_volatile`](https://en.cppreference.com/w/cpp/types/is_volatile)
    * [`is_trivial`](https://en.cppreference.com/w/cpp/types/is_trivial)
    * [`is_trivially_copyable`](https://en.cppreference.com/w/cpp/types/is_trivially_copyable)
    * [`is_standard_layout`](https://en.cppreference.com/w/cpp/types/is_standard_layout)
    * [`has_unique_object_representations`](https://en.cppreference.com/w/cpp/types/has_unique_object_representations)
    * [`is_empty`](https://en.cppreference.com/w/cpp/types/is_empty)
    * [`is_polymorphic`](https://en.cppreference.com/w/cpp/types/is_polymorphic)
    * [`is_abstract`](https://en.cppreference.com/w/cpp/types/is_abstract)
    * [`is_final`](https://en.cppreference.com/w/cpp/types/is_final)
    * [`is_aggregate`](https://en.cppreference.com/w/cpp/types/is_aggregate)
    * [`is_unsigned`](https://en.cppreference.com/w/cpp/types/is_unsigned)
    * [`is_bounded_array`](https://en.cppreference.com/w/cpp/types/is_bounded_array)
    * [`is_unbounded_array`](https://en.cppreference.com/w/cpp/types/is_unbounded_array)
    * [`is_scoped_enum`](https://en.cppreference.com/w/cpp/types/is_scoped_enum)
    * [`rank`](https://en.cppreference.com/w/cpp/types/rank)
    
3. **Type aliases of `std::type_traits`.** These are conditionally supported. Since types are parsed differently from values, the `.member-trait-name` syntax applies like a [_ptr-operator_](http://eel.is/c++draft/dcl.decl.general#nt:ptr-operator). It has left-to-right associativity. Chain them together to perform complex type transformations. Note that if you want to apply a type alias member trait to a declarator that has a [_noptr-operator_](http://eel.is/c++draft/dcl.decl.general#nt:noptr-declarator), you'll need to use C++'s quite obscure [spiral rule](https://stackoverflow.com/questions/16260417/the-spiral-rule-about-declarations-when-is-it-in-error) for declarations. `char(.decay)[3]` decays the type `char[3]`. We can't simply write `char[3].decay`, because that is ungrammatical in C++.
    * [`remove_cv`](https://en.cppreference.com/w/cpp/types/remove_cv)
    * [`remove_const`](https://en.cppreference.com/w/cpp/types/remove_const)
    * [`remove_volatile`](https://en.cppreference.com/w/cpp/types/remove_volatile)
    * [`add_cv`](https://en.cppreference.com/w/cpp/types/add_cv)
    * [`add_const`](https://en.cppreference.com/w/cpp/types/add_const)
    * [`add_volatile`](https://en.cppreference.com/w/cpp/types/add_volatile)
    * [`remove_reference`](https://en.cppreference.com/w/cpp/types/remove_reference)
    * [`add_lvalue_reference`](https://en.cppreference.com/w/cpp/types/add_lvalue_reference)
    * [`add_rvalue_reference`](https://en.cppreference.com/w/cpp/types/add_rvalue_reference)
    * [`remove_pointer`](https://en.cppreference.com/w/cpp/types/remove_pointer)
    * [`add_pointer`](https://en.cppreference.com/w/cpp/types/add_pointer)
    * [`make_signed`](https://en.cppreference.com/w/cpp/types/make_signed)
    * [`make_unsigned`](https://en.cppreference.com/w/cpp/types/make_unsigned)
    * [`remove_extent`](https://en.cppreference.com/w/cpp/types/remove_extent)
    * [`remove_all_extents`](https://en.cppreference.com/w/cpp/types/remove_all_extents)
    * [`decay`](https://en.cppreference.com/w/cpp/types/decay)
    * [`remove_cvref`](https://en.cppreference.com/w/cpp/types/remove_cvref)
    * [`underlying_type`](https://en.cppreference.com/w/cpp/types/underlying_type)

