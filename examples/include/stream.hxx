#include <string>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include "util.hxx"

template<typename type_t>
void stream(std::ostream& os, const type_t& obj, int indent) {

  os<< @type_name(type_t)<< " ";
  int indent_next = -1 != indent ? indent + 1 : -1;

  if constexpr(is_spec_t<std::basic_string, type_t>::value) {
    // Carve out an exception for strings. Put the text of the string
    // in quotes. We could go further and add character escapes back in.
    os<< '\"'<< obj<< '\"';

  } else if constexpr(is_spec_t<std::vector, type_t>::value) {
    // Special treatment for std::vector. Output each element in a comma-
    // separated list in brackets.
    os<< '[';
    bool insert_comma = false;
    for(const auto& x : obj) {
      // Move to the next line and indent.
      if(insert_comma)
        os<< ',';

      if(-1 != indent)
        os<< "\n"<< std::string(2 * (indent + 1), ' ');
      else
        os<< ' ';
      
      // Stream the element.
      stream(os, x, indent_next);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }

    if(-1 != indent)
      os<< "\n"<< std::string(2 * indent, ' ')<< "]";
    else
      os<< " ]";

  } else if constexpr(is_spec_t<std::map, type_t>::value) {
    // Special treatment for std::map.
    os<< "{";
    bool insert_comma = false;
    for(const auto& x : obj) {
      if(insert_comma)
        os<< ",";

      if(-1 != indent)
        os<< "\n"<< std::string(2 * (indent + 1), ' ');
      else
        os<< ' ';

      // stream the key.
      stream(os, x.first, indent_next);

      os<< " : ";

      // stream the value.
      stream(os, x.second, indent_next);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }  
    if(-1 != indent)
      os<< "\n"<< std::string(2 * indent, ' ')<< "}";
    else
      os<< " }";

  } else if constexpr(is_spec_t<std::optional, type_t>::value) {
    // For an optional member, either stream the value or stream "null".
    if(obj.has_value())
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
      
      if(-1 != indent) 
        os<< "\n"<< std::string(2 * (indent + 1), ' ');
      else
        os<< ' ';

      // Stream the name of the member. The type will be prefixed before the
      // value.
      os<< @member_name(type_t, i)<< " : ";

      // Stream the value of the member.
      stream(os, @member_ref(obj, i), indent_next);

      // On the next go-around, insert a comma before the newline.
      insert_comma = true;
    }
    if(-1 != indent)
      os<< "\n"<< std::string(2 * indent, ' ')<< "}";
    else
      os<< " }";

  } else {
    // For any non-class type, use the iostream overloads.
    os<< '\"'<< obj<< '\"';
  }
}