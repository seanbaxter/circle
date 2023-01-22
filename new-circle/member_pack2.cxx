#include <iostream>

template<typename T, int N>
struct soa {
  [[member_names(T~member_names...)]] T~member_types ...m[N];

  T get_object(int index) const {
    // Use the pack name in member functions: this is generic.
    return { m[index]... };
  }

  void set_object(int index, T obj) {
    // Decompose obj and write into component arrays.
    m[index] = obj~member_values ...;
  }
};

struct vec3_t { float x, y, z; };
using my_soa = soa<vec3_t, 8>;

int main() {
  std::cout<< my_soa~member_decl_strings + "\n" ...; 

  my_soa obj;
  obj.x[0] = 1;   // Access member names cloned from vec3_t
  obj.y[1] = 2;
  obj.z[2] = 3;
}