#include <cstdio>
#include <string>
#include <stdexcept>

template<typename type_t>
std::string enum_to_string(type_t e) {
  switch(e) {
    // A compile-time loop inside a switch.
    @meta for enum(type_t e2 : type_t) {
      case e2: 
        return @enum_name(e2);
    } 
    default:
      return "<unknown>";
  }
}

int main() {

  enum class shape_t {
    circle,
    square, 
    triangle,
    hexagon,
  };

  struct foo_t {
    double radius;
    shape_t shape;
  };

  foo_t obj { 3.0, shape_t::triangle };

  const char* s = @match(obj) {
    // Compile-time loop over each enum in shape_t.
    @meta for enum(auto shape : shape_t) {

      // Compile-time loop over the pairs in this array. Use structured
      // bindings for r_limit and size.
      @meta std::pair<double, std::string> sizes[] {
        { 1.0, "small" }, { 5.0, "medium" }, { -1, "large" }
      };
      
      @meta for(auto& [r_limit, size] : sizes) {
        // Form an std::string concatenating our message as a compile-time 
        // object. Use @string to convert it to a string literal available
        // at runtime.
        @meta std::string s = "A " + size + " " + enum_to_string(shape);
        @meta printf("%s\n", s.c_str());

        // If -1 != r_limit, test against the radius and the shape. Otherwise
        // test only against the shape.
        @meta if(-1 != r_limit)
          // Match only when radius < r_limit.
          [.radius: < r_limit, .shape: shape] => @string(s);
        else
          [.shape: shape] => @string(s);
      }
    }

    // Create a default.
    _ => "Unrecognized shape";
  };

  printf("%s\n", s);
  return 0;
}