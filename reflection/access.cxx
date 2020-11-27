#include <cstdio>

struct vec3_t {
public:    float x;
protected: float y;
private:   float z;
};

template<typename type_t, int access>
void print_members() {
  static_assert(access >= 0 && access <= 7);
  printf("access = %d:\n", access);
  printf("  %2d: %-10s (offset %2d)\n",
    int..., 
    @member_decl_strings(type_t, access), 
    @member_offsets(type_t, access)
  )...;
}

int main() {
  // Use a template parameter for access protection.
  @meta for(int access = 0; access <= 7; ++access)
    print_members<vec3_t, access>();
}