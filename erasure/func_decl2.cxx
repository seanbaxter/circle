#include <cstdio>

template<typename interface>
struct my_base_t {
  // Loop over each method in interface_t.
  @meta for(int i = 0; i < @method_count(interface); ++i) {
    @meta printf("Injecting %s: %s\n", 
      @method_name(interface, i), 
      @type_name(@method_type(interface, i))
    );

    // Declare a pure virtual function with the same name and signature.
    virtual @func_decl(@method_type(interface, i), @method_name(interface, i), args) = 0;
  }
};

struct my_interface_t {
  void print(const char* text);
  bool save(const char* filename, const char* data);
  void close();
};


int main() {
  typedef my_base_t<my_interface_t> my_base;

  // Print all the method names and types:
  @meta for(int i = 0; i < @method_count(my_base); ++i) {
    @meta printf("Found %s: %s\n", 
      @method_name(my_base, i), 
      @type_name(@method_type(my_base, i))
    );
  }

  return 0;
}