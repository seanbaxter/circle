#include <tuple>

// Declare a tuple specialization.
using T = std::tuple<char, int, void*, const char*, float, int[5]>;

// Create a new specialization of the same template, but with the
// arguments sorted in lexicographical order.
using T2 = T.template<T.type_args.sort(_0.string < _1.string)...>;

// Confirm the new arguments are sorted!
static_assert(T2 == std::tuple<char, const char*, float, int, int[5], void*>);