#include <tuple>
#include <cstddef>

// Test on a couple cases.
using T1 = std::pair<float, double>;
using T2 = std::tuple<char, short, int, long>;

static_assert(double == T1.type_args...[1]);
static_assert(char   != T1.type_args...[0]);

static_assert(int    == T2.type_args...[2]);
static_assert(double != T2.type_args...[1]);