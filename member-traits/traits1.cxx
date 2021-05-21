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
