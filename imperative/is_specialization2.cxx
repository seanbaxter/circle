#include <tuple>

using T1 = std::tuple<int, char>;
using T2 = std::pair<int, char>;

static_assert(T1.template == std::tuple);
static_assert(T2.template != std::tuple);
