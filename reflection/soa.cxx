#include <vector>
#include <sstream>
#include <cstdio>

// Define a struct that has a std::vector for each member of type_t.
template<typename type_t>
struct vectors_t {
  std::vector<@member_types(type_t)> @("vec_", @member_names(type_t)) ...;
};

// Perform AOS to SOA by copying data members into the vectors.
template<typename type_t>
vectors_t<type_t> aos_to_soa(const type_t* objects, size_t count) {
  vectors_t<type_t> vec;
  vec.@member_values().resize(count)...;

  for(size_t i = 0; i < count; ++i)
    vec.@member_values()[i] = objects[i].@member_values()...;

  return vec;
}

template<typename type_t>
std::string print_vector(const std::vector<type_t>& vec) {
  std::ostringstream oss;
  oss<< "[";
  if(vec.size()) {
    oss<< " "<< vec[0];
    oss<< ", "<< vec[1:]...;
  }
  oss<< " ]";
  return oss.str();
}

template<typename type_t>
void print_vectors(const type_t& vecs) {
  printf("%s\n", @type_string(type_t));
  printf("  %s: %s\n", @member_names(type_t), 
    print_vector(vecs.@member_values()).c_str())...;
}

struct vec3_t {
  double x, y, z;
};

int main() {
  vec3_t data[] {
    1.0, 2.0, 3.0,
    1.1, 2.1, 3.1,
    1.2, 2.2, 3.2,
    1.3, 2.3, 3.3
  };

  auto soa = aos_to_soa(data, data.length);

  print_vectors(soa);
}