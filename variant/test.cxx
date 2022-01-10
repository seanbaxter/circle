#include "variant.hxx"

#include <string>
#include <iostream>

struct no_default_ctor { 
  no_default_ctor() = delete; 
};

struct throw_default_ctor {
  throw_default_ctor() noexcept(false) { }
};

struct no_copy_ctor {
  no_copy_ctor(const no_copy_ctor&) = delete;
};

struct no_move_ctor {
  no_move_ctor(no_move_ctor&&) = delete;
};

struct no_assignment { 
  no_assignment& operator=(const no_assignment&) = delete; 
};

using namespace circle;

// Do some type-trait tests.
static_assert(std::is_default_constructible_v<variant<long, void*>>);
static_assert(std::is_nothrow_default_constructible_v<variant<long, void*>>);

static_assert(std::is_default_constructible_v<variant<long, std::string>>);
static_assert(std::is_nothrow_default_constructible_v<variant<std::string, long>>);
static_assert(!std::is_nothrow_default_constructible_v<variant<throw_default_ctor, long>>);

static_assert(std::is_default_constructible_v<variant<long, no_default_ctor>>);
static_assert(!std::is_default_constructible_v<variant<no_default_ctor, long>>);

static_assert(std::is_copy_constructible_v<variant<std::string, long>>);
static_assert(!std::is_nothrow_copy_constructible_v<variant<std::string, long>>);
static_assert(!std::is_copy_constructible_v<variant<std::string, no_copy_ctor>>);

static_assert(std::is_trivially_move_constructible_v<variant<long, void*>>);
static_assert(std::is_move_constructible_v<variant<std::string, long>>);
static_assert(!std::is_trivially_move_constructible_v<variant<std::string, long>>);
static_assert(!std::is_move_constructible_v<variant<no_move_ctor, long>>);

static_assert(std::is_trivially_copy_assignable_v<variant<long, void*>>);
static_assert(std::is_copy_assignable_v<std::string>);
static_assert(std::is_copy_assignable_v<variant<std::string, long>>);
static_assert(!std::is_trivially_copy_assignable_v<std::string>);
static_assert(!std::is_trivially_copy_assignable_v<variant<std::string, long>>);

static_assert(!std::is_copy_assignable_v<variant<no_assignment, long>>);

// Allow string construction from a const char*.
static_assert(std::is_constructible_v<variant<std::string>, const char*>);

// But fail when there are multiple variant members due to ambigious
// __preferred_copy_init.
static_assert(!std::is_constructible_v<variant<std::string, std::string>, const char*>);

// Same with assignment
static_assert(std::is_assignable_v<variant<std::string>&, const char*>);
static_assert(!std::is_assignable_v<variant<std::string, std::string>&, const char*>);

int main() {
  using namespace circle;

  // Default construct.
  variant<int, long> v1;
  variant<std::string, long> v2;
  
  {
    // Copy construct.
    auto v3 = v1;
    auto v4 = v2;
  }
  
  {
    // Move construct.
    auto v3 = std::move(v1);
    auto v4 = std::move(v2);
  }

  // Converting constructor.
  variant<std::string, long> v3 = "Hello world";
  variant<std::string, long> v4 = 1001;

  // Getter.
  std::cout<< get<0>(v3)<< " "<< get<std::string>(v3)<< "\n";
  std::cout<< get<1>(v4)<< " "<< get<long>(v4)<< "\n";

  // Visitor.
  visit([](auto a, auto b) {
    std::cout<< a<< " "<< b<< "\n";
  }, v3, v4);
}
