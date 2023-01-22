#pragma feature choice
#include <type_traits>

struct A {
  // Declare a non-trivial destructor to keep things interesting.
  ~A() { }

  // Declare potentially-throwing copy constructor.
  A(const A&) noexcept(false);

  // We must define a non-throwing move constructor.
  A(A&&) noexcept;

  // Define a move assignment operator, because [class.copy.assign]/4 
  // prevents is generation.
  A& operator=(A&&) noexcept;
};

choice my_choice_t {
  // We need a choice type with at least two alternatives to get into
  // a code path that calls copy constructors during choice assignment.
  value(A),
  value2(int),
};

// The choice type is *not* copy-assignable, because that could leave it in
// a valueles-by-exception state.
static_assert(!std::is_copy_assignable_v<my_choice_t>);

// However, it *is* move-assignable.
static_assert(std::is_move_assignable_v<my_choice_t>);

void copy_assign(my_choice_t& lhs, const my_choice_t& rhs) {
  // Simulate copy-assignment in 2 steps:
  // 1. Copy-construct the rhs.
  // 2. Move-assign that temporary into the lhs.
  // lhs = rhs;            // ERROR!
  lhs = my_choice_t(rhs);  // OK!
}