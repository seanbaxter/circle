#include <type_traits>
#include <tuple>

// We can't return arrays by value, so take the argument by reference.
template<typename type_t>
void subgroupShuffle(type_t& x, uint id) {
  if constexpr(
    std::is_array_v<type_t> || 
    __is_vector(type_t) || 
    __is_matrix(type_t) || 
    requires { typename std::tuple_size<type_t>::type; }) {

    // Shuffle elements of arrays, vectors, matrices and tuples.
    subgroupShuffle(x.[:], id)...;

  } else if constexpr(std::is_class_v<type_t>) {
    // Shuffle all public base classes and data members of class objects.
    subgroupShuffle(x.@base_values, id)...;
    subgroupShuffle(x.@member_values, id)...;

  } else {
    // Plain shuffle scalars.
    x = gl_subgroupShuffle(x, id);
  }
}

// Overload the SPIR-V declaration gl_subgroupShuffle.
template<typename type_t>
type_t gl_subgroupShuffle(type_t x, uint id) {
  subgroupShuffle(x, id);
  return x;
}

// Create a complex test case with inheritance, arrays and a tuple.
struct base_t {
  double d[2];
};

struct box_t {
  vec3 min, max;
};

struct foo_t : base_t {
  mat4 m;
  vec4 v[2];
  std::tuple<int, float, int> tuple;
  box_t box;
};

[[spirv::buffer(0)]]
foo_t foos[];

extern "C" [[spirv::comp, spirv::local_size(128)]]
void comp() {
  int gid = glcomp_GlobalInvocationID.x;

  // Each thread loads a foo.
  foo_t foo = foos[gid];

  // Broadcast lane 3's foo.
  foo = gl_subgroupShuffle(foo, 3);

  // Store back to mem.
  foos[gid] = foo;
}