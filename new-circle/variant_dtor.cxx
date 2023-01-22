#include <type_traits>
#include <limits>
#include <vector>
#include <string>
#include <memory>

static constexpr size_t variant_npos = size_t~max;

template<class... Types>
class variant {
  static constexpr bool trivially_destructible = 
    (... && Types~is_trivially_destructible);

  union { Types ...m; };
  uint8_t _index = variant_npos;

public:
  // Conditionally define the default constructor.
  constexpr variant() 
  noexcept(Types...[0]~is_nothrow_default_constructible) 
  requires(Types...[0]~is_default_constructible) : 
    m...[0](), _index(0) { }

  // Conditionally-trivial destructor.
  constexpr ~variant() requires(trivially_destructible) = default;
  constexpr ~variant() requires(!trivially_destructible) { reset(); }

  constexpr void reset() noexcept {
    if(_index != variant_npos) {
      int... == _index ...? m.~Types() : __builtin_unreachable();
      _index = variant_npos;        // set to valueless by exception.
    }
  }
};

int main() {
  // Instantiate the variant so that the destructor is generated.
  variant<std::string, std::vector<int>, double, std::unique_ptr<double>> vec;
}