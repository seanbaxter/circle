#include <cstdio>

template<auto x, typename type_t>
constexpr bool is_value_in_enum = (... || (@enum_values(type_t) == (type_t)x));

// Set of device architectures. Will be part of CUDA Toolkit.
enum class sm_selector : unsigned long long {
  sm_35 = 35, sm_37 = 37,
  sm_50 = 50, sm_52 = 52, sm_53 = 53,
  sm_60 = 60, sm_61 = 61, sm_62 = 62,
  sm_70 = 70, sm_72 = 72, sm_75 = 75,
  sm_80 = 80, sm_86 = 86,
};

// tuning params
using nt  [[attribute   ]] = int;
using vt  [[attribute(1)]] = int;
using occ [[attribute(0)]] = int;

// flags
using strided    [[attribute]] = void;
using persistent [[attribute]] = void;

// Tunings for a specific operation.
enum class tuning_t {
  kepler  [[ .nt=128, .vt=5               ]] = 35,
  maxwell [[ .nt=256, .vt=7,  .persistent ]] = 52,
  pascal  [[ .nt=64,  .vt=11, .strided    ]] = 61,
  turing  [[ .nt=256, .vt=15, .occ=3      ]] = 75,
  ampere  [[ .nt=256, .vt=19, .strided    ]] = 86,
};

// Test that each tuning corresponds to an actual device architecture.
static_assert(
  is_value_in_enum<@enum_values(tuning_t), sm_selector>,
  @string(@enum_names(tuning_t), " (", (int)@enum_values(tuning_t), ") is invalid")
)...;

int main() { 
  // Print the tunings using a loop.
  printf("With a loop:\n");
  @meta for enum(tuning_t tuning : tuning_t) {
    printf("%-10s: %3dx%2d\n", @enum_name(tuning),
      @enum_attribute(tuning, nt), @enum_attribute(tuning, vt));
  }

  // Print the tunings using pack expansion.
  printf("\nWith a pack expansion:\n");
  printf("%-10s: %3dx%2d\n", @enum_names(tuning_t), 
    @enum_attributes(tuning_t, nt), @enum_attributes(tuning_t, vt)) ...;
}
