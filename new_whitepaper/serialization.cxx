#include <iostream>
#include <string>

template<typename type_t>
concept C1 = std::is_class<type_t>::value;

template<C1 type_t>
void print_obj(const type_t& obj) {
  // Loop over each non-static data member.
  @meta for(int i = 0; i < @member_count(type_t); ++i) {
    // Print its member name and member value.
    std::cout<< @member_name(type_t, i)<< " : "<< @member_ref(obj, i)<< "\n";
  }
}

struct foo_t {
  int a;
  double b;
  const char* c;
  std::string d;
};

int main() {
  foo_t foo {
    5,
    3.14,
    "A C string",
    "A C++ string"
  };

  print_obj(foo);
  return 0;
}