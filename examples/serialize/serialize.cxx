// Use introspect to iterate over each member of a class and
// stream its output.

#include <iostream>
#include <type_traits>

template<typename type_t>
void stream(std::ostream& os, const type_t& obj) {
  // @member_name won't work on non-class types, so check that here.
  static_assert(std::is_class<type_t>::value, "stream requires class type");

  // Stream the type name followed by the object name.
  os<< @type_name(type_t)<< " {\n";

  // Iterate over each member of type_t.
  @meta for(int i = 0; i < @member_count(type_t); ++i) {
    // Stream the member name and the member value.
    os<< "  "<< 
      @type_name(@member_type(type_t, i))<< " "<< 
      @member_name(type_t, i)<< ": "<<
      <<'\"'<< @member_ref(obj, i)<< "\"\n";
  }
  os<<"}\n";
}

struct struct1_t {
  char c;
  double d;
  const char* s;
};

struct struct2_t {
  std::string string;
  long l;
  bool b;
};

int main(int argc, char** argv) {
  struct1_t one {
    'X',
    3.14159,
    "A C string"
  };
  stream(std::cout, one);

  struct2_t two {
    "A C++ string",
    42,
    true
  };
  stream(std::cout, two);
  
  return 0;
}
