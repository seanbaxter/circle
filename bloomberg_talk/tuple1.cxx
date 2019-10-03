#include <iostream>

template<typename... types_t>
struct tuple_t {
  @meta for(int i = 0; i < sizeof...(types_t); ++i)
    types_t...[i] @(i);
};

template<typename type_t>
void print_object(const type_t& obj) {
  std::cout<< 
    @type_name(@member_types(type_t))<< " "<< 
    @member_names(type_t)<< ": "<< 
    @member_pack(obj)<< "\n" ...;
}

@meta print_object(tuple_t<int, std::string, double> {
  5, "Hello tuple", 1.618
});

int main() {
  return 0;
}