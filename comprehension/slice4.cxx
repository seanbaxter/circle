#include <iostream>

template<typename... types_t>
struct tuple_t {
  types_t @(int...) ...;
};

template<typename... types_t>
tuple_t(types_t... args) -> tuple_t<types_t...>;

template<typename type_t>
void print_object(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int... << ") "<< 
    decltype(obj.[:]).string<< " : "<< 
    obj.[:]<< "\n" ...;
}

template<typename type_t>
void print_reverse(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int... << ") "<< 
    decltype(obj.[::-1]).string<< " : "<< 
    obj.[::-1]<< "\n" ...;
}

template<typename type_t>
void print_odds(const type_t& obj) {
  std::cout<< @type_string(type_t)<< "\n";
  std::cout<< "  "<< int... << ") "<< 
    decltype(obj.[1::2]).string<< " : "<< 
    obj.[1::2]<< "\n" ...;
}

int main() {
  tuple_t obj { 3.14, 100l, "Hi there", "Member 3", 'q', 19u };

  std::cout<< "print_object:\n";
  print_object(obj);

  std::cout<< "\nprint_reverse:\n";
  print_reverse(obj);

  std::cout<< "\nprint_odds:\n";
  print_odds(obj);
}
