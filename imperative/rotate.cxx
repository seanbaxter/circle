#include <tuple>

template<size_t N, typename T>
using Rotate = T.template<
  T.universal_args...[N:]..., T.universal_args...[:N]...
>;

using T1 = std::tuple<char, short, int, long, long long>;

static_assert(std::tuple<int, long, long long, char, short> == Rotate<2, T1>);