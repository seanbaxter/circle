#include <cstdio>

template<typename type1_t, typename type2_t>
auto dot_product(const type1_t& left, const type2_t& right) {
  return ((@member_pack(left) * @member_pack(right)) + ...);
}

template<typename type_t>
auto norm(const type_t& obj) {
  return sqrt(dot_product(obj, obj));
}

struct vec3_t {
  float x, y, z;
};

struct vec4_t {
  float x, y, z, w;
};

int main() {
  vec3_t vec3 { 1, 2, 3 };
  float mag3 = norm(vec3);
  printf("mag3 = %f\n", mag3);

  vec4_t vec4 { 1, 2, 3, 4 };
  float mag4 = norm(vec4);
  printf("mag4 = %f\n", mag4);

  // Uncomment this to attempt the dot product of a vec3_t and vec4_t.e
  // float dot = dot_product(vec3, vec4);

  return 0;
}