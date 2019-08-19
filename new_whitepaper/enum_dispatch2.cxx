#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <type_traits>

enum class shape_t {
  circle,
  triangle,
  square, 
  hexagon, 
  octagon,
};

double circle(double r) {
  return M_PI * r * r;
}

double triangle(double r) {
  return r * sqrt(3) / 2;
}

double square(double r) {
  return 4 * r * r;
}

double hexagon(double r) {
  return 1.5 * sqrt(3) * (r * r);
}

double octagon(double r) {
  return 2 * sqrt(2) * (r * r);
}

double polygon_area(shape_t shape, double r) {
  switch(shape) {
    @meta for enum(shape_t s : shape_t) {
      case s: 
        // Call the function that has the same name as the enumerator.
        return @(@enum_name(s))(r);
    }

    default: 
      assert(false);
      return 0;
  }
}

template<typename type_t>
std::string enum_to_string(type_t x) {
  static_assert(std::is_enum<type_t>::value);

  switch(x) {
    @meta for enum(auto y : type_t)
      case y: return @enum_name(y);

    default: return "<" + std::to_string((int)x) + ">";
  }
}

template<typename type_t>
type_t string_to_enum(const char* name) {
  static_assert(std::is_enum<type_t>::value);

  @meta for enum(type_t x : type_t) {
    if(!strcmp(name, @enum_name(x)))
      return x;
  }

  throw std::runtime_error("value is not an enumerator");

  printf("%s is not an enumerator of type %s\n", name, @type_name(type_t));
  exit(1);
}

const char* usage = "  enums3 <shape-name> <radius>\n";

int main(int argc, char** argv) {
  if(3 != argc) {
    puts(usage);
    exit(1);
  }

  shape_t shape = string_to_enum<shape_t>(argv[1]);
  double radius = atof(argv[2]);

  double area = polygon_area(shape, radius);

  printf("Area of %s of radius %f is %f\n", enum_to_string(shape).c_str(),
    radius, area);

  return 0;
}


