#include <tuple>

template<size_t N, typename T>
using Rotate = T.template<
  auto Count : sizeof...T.universal_args =>
    for i : Count =>
      T.universal_args...[(i + N) % Count]
>;

using T1 = std::tuple<char, short, int, long, long long>;

static_assert(std::tuple<int, long, long long, char, short> == Rotate<2, T1>);