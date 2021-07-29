#include <tuple>
#include <variant>

template<typename T1, typename T2>
using Rebind = T1.template<T2.universal_args...>;

using T1 = std::tuple<>;
using T2 = std::variant<int, double, char, float>;
using T3 = std::tuple<int, double, char, float>;

// Use the alias template.
static_assert(T3 == Rebind<T1, T2>);

// Do it in situ.
static_assert(T3 == T1.template<T2.universal_args...>);