#include <iostream>

template<typename type_t>
void print_object(const type_t& obj) {
  std::cout<< 
    @type_name(@member_types(type_t))<< " "<< 
    @member_names(type_t)<< ": "<< 
    @member_pack(obj)<< "\n" ...;
}

template<typename type_t>
struct tuple_t {
  static_assert(__is_typed_enum(type_t), 
    "template argument to tuple_t must be typed enum");

  @meta for enum(type_t e : type_t)
    @enum_type(e) @(@enum_name(e));
};

enum typename class my_list_t {
  Int = int,
  Double = double,
  String = std::string,
  Array = float[4],
  Int2 = int,
};

int main() {

  tuple_t<my_list_t> my_tuple {
    5, 3.14, "A list tuple", { 4, 5, 6, 7 }, 3
  };
  print_object(my_tuple);

  return 0;
}