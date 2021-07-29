#include <tuple>
#include <type_traits>
#include <cstddef>

// Define a class template to get the index'th type pack element.
template<size_t index, typename T, typename... Ts> 
struct get_pack_arg {
  using type = typename get_pack_arg<index - 1, Ts...>::type;
};

template<typename T, typename... Ts> 
struct get_pack_arg<0, T, Ts...> {
  using type = T;
};

// Define a class template to expose a specialization's template arguments
// using argument deduction.
template<size_t index, typename T> 
struct get_arg;

template<size_t index, template<typename...> class Temp, typename... Ts>
struct get_arg<index, Temp<Ts...> > {
  using type = typename get_pack_arg<index, Ts...>::type;
};

// Define an alias template for ergonomics.
template<size_t index, typename T>
using get_arg_t = typename get_arg<index, T>::type;

// Test on a couple cases.
using T1 = std::pair<float, double>;
using T2 = std::tuple<char, short, int, long>;

static_assert(std::is_same_v<double, get_arg_t<1, T1> >);
static_assert(!std::is_same_v<char, get_arg_t<0, T1> >);

static_assert(std::is_same_v<int, get_arg_t<2, T2> >);
static_assert(!std::is_same_v<double, get_arg_t<1, T2> >);