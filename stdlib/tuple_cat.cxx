#include <tuple>
#include <string>
#include <iostream>

template<class... Tuples>
constexpr std::tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat1(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      auto N : Ti.remove_reference.tuple_size =>
        get<int...(N)>(std::forward<Ti>(tpls...[i]))...
  };
}

template<class... Tuples>
constexpr std::tuple<
  for typename Ti : Tuples => 
    Ti.remove_reference.tuple_elements...
>
tuple_cat2(Tuples&&... tpls) {
  return { 
    for i, typename Ti : Tuples =>
      std::forward<Ti>(tpls...[i]) ...
  };
}

int main() {
  using namespace std::string_literals;
  auto t1 = std::make_tuple(1, 2.2, "Three");
  auto t2 = std::make_tuple("Four"s, 5i16);
  auto t3 = std::make_tuple(6.6f, 7ull);

  auto cat  = std::tuple_cat(t1, t2, t3);
  auto cat1 = tuple_cat1(t1, t2, t3);
  auto cat2 = tuple_cat2(t1, t2, t3);
  
  std::cout<< "cat == cat1 is "<< (cat == cat1 ? "true\n" : "false\n");
  std::cout<< "cat == cat2 is "<< (cat == cat2 ? "true\n" : "false\n");

  std::cout<< decltype(cat2).tuple_elements.string<< ": "<< cat2.[:]<< "\n" ...;  
}