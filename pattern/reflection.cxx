#include <iostream>
#include <concepts>

using namespace std;

template<typename T>
concept small_type = T is void || sizeof(T) <= 4;

// The return type must satisfy std::integral.
// All parameter types must satisfy std::floating_point.
template<typename F>
constexpr bool P = 
  { F.return_type, F.param_types... } is [ small_type, ...floating_point ];

static_assert(true == P<void(float, float)>);
static_assert(false == P<void(short, float, float, double)>);
static_assert(true == P<char(double, float, float, double)>);

// Define a type to associate with each enumerator.
using assoc_type [[attribute]] = typename;

enum class shapes_t {
  circle    [[.assoc_type=float ]],
  ellipse   [[.assoc_type=double]],
  square    [[.assoc_type=int   ]],
  rectangle [[.assoc_type=long  ]],
};

// Check that the assoc_type user-defined attributes for the enumerators in
// shapes_t matches an expected pattern.
static_assert({ @enum_tattributes(shapes_t, assoc_type)... } is
  [float, double, int, long]);

// Check that the assoc_type are all scalars.
static_assert({ @enum_tattributes(shapes_t, assoc_type)... } is
  [... is_scalar_v]);

// Check that they are all small types. They aren.t
static_assert({ @enum_tattributes(shapes_t, assoc_type)... } is not 
  [... small_type]);