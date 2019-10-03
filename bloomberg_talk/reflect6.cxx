#include <iostream>
#include <type_traits>
#include <map>
#include <vector>
#include <string>
#include <optional>


template<typename type_t>
const char* enum_to_name(type_t e) {
  switch(e) {
    @meta for enum(type_t e2 : type_t) {
      case e2:
        return @enum_name(e2);
    }

    default:
      return nullptr;
  }
}

template<typename type_t>
void stream(std::ostream& os, const type_t& obj, int indent) {

  os<< @type_name(type_t)<< " ";

  if constexpr(std::is_enum<type_t>::value) {
    os<< '\"';
    if(const char* name = enum_to_name<type_t>(obj))
      // Write the enumerator name if the value maps to an enumerator.
      os<< name;
    else
      // Otherwise cast the enum to its underlying type and write that.
      os<< (typename std::underlying_type<type_t>::type)obj;
    os<< '\"';

  } else if constexpr(@is_class_template(type_t, std::basic_string)) {
    // Carve out an exception for strings. Put the text of the string
    // in quotes. We could go further and add character escapes back in.
    os<< '\"'<< obj<< '\"';

  } else if constexpr(@is_class_template(type_t, std::vector)) {
    // Special treatment for std::vector. Output each element in a comma-
    // separated list in brackets.
    os<< "[";
    bool insert_comma = false;
    for(const auto& x : obj) {
      // Move to the next line and indent.
      if(insert_comma)
        os<< ',';
      os<< "\n"<< std::string(2 * (indent + 1), ' ');
      
      // Stream the element.
      stream(os, x, indent + 1);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< "\n"<< std::string(2 * indent, ' ')<< "]";

  } else if constexpr(@is_class_template(type_t, std::map)) {
    // Special treatment for std::map.
    os<< "{";
    bool insert_comma = false;
    for(const auto& x : obj) {
      if(insert_comma)
        os<< ",";
      os<< "\n"<< std::string(2 * (indent + 1), ' ');

      // Stream key : value, where the key and value are done recursively.
      stream(os, x.first, indent + 1);
      os<< " : ";
      stream(os, x.second, indent + 1);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }  
    os<< "\n"<< std::string(2 * indent, ' ')<< "}";

  } else if constexpr(@is_class_template(type_t, std::optional)) {
    // For an optional member, either stream the value or stream "null".
    if(obj)
      stream(os, *obj, indent);
    else
      os<< "null";

  } else if constexpr(std::is_class<type_t>::value) {
    // For any other class, treat with circle's introspection.
    os<< "{";
    bool insert_comma = false;
    @meta for(size_t i = 0; i < @member_count(type_t); ++i) {
      if(insert_comma) 
        os<< ",";
      os<< "\n"<< std::string(2 * (indent + 1), ' ');

      // Stream the name of the member. The type will be prefixed before the
      // value.
      os<< @member_name(type_t, i)<< " : ";

      // Stream the value of the member.
      stream(os, @member_ref(obj, i), indent + 1);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    os<< "\n"<< std::string(2 * indent, ' ')<< "}";

  } else {
    // For any non-class type, use the iostream overloads.
    os<< '\"'<< obj<< '\"';
  }
}

struct vec3_t {
  double x, y, z;
};
typedef std::map<std::string, vec3_t> vec_map_t;

enum class robot_t {
  T800,
  R2D2,
  RutgerHauer,
  Mechagodzilla,
  Bishop,
};

struct struct1_t {
  std::string s;
  std::vector<int> a;
  vec3_t vec;
  robot_t r1, r2;
  vec_map_t axes;
  std::optional<int> opt_1;
  std::optional<vec3_t> opt_2;
  int x;
};

int main() {
  struct1_t obj { };
  obj.s = "struct1_t instance";
  obj.a.push_back(4);
  obj.a.push_back(5);
  obj.a.push_back(6);
  obj.vec = vec3_t { 1, 2, 3 };
  obj.r1 = robot_t::R2D2;
  obj.r2 = robot_t::RutgerHauer;
  obj.axes["x"] = vec3_t { 1, 0, 0 };
  obj.axes["y"] = vec3_t { 0, 1, 0 };
  obj.axes["z"] = vec3_t { 0, 0, 1 };
  obj.opt_1 = 500;
  // Don't set opt_2.
  obj.x = 600;

  stream(std::cout, obj, 0);
  std::cout<< std::endl;
  return 0;
}