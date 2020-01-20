#include <iostream>

template<int... x, typename... types_t>
void func1(types_t... args) {
  std::cout<< "Expansion expression of static parameter pack:\n";

  std::cout<< "  Non-type template parameters:\n";
  std::cout<< "    "<< x<<"\n" ...;

  std::cout<< "  Type template parameters:\n";
  std::cout<< "    "<< @type_string(types_t)<< "\n" ...;

  std::cout<< "  Function template parameters:\n";
  std::cout<< "    "<< args<< "\n" ...;
}

template<int... x, typename... types_t>
void func2(types_t... args) {
  std::cout<< "\nReverse-order direct pack indexing with ...[index]:\n";

  std::cout<< "  Non-type template parameters:\n";
  @meta for(int i = sizeof...(x) - 1; i >= 0; --i)
    std::cout<< "    "<< x...[i]<<"\n";

  std::cout<< "  Type template parameters:\n";
  @meta for(int i = sizeof...(x) - 1; i >= 0; --i)
    std::cout<< "    "<< @type_string(types_t...[i])<< "\n";

  std::cout<< "  Function template parameters:\n";
  @meta for(int i = sizeof...(x) - 1; i >= 0; --i)
    std::cout<< "    "<< args...[i]<< "\n";
}

template<int... x, typename... types_t>
void func3(types_t... args) {
  std::cout<< "\nReverse-order pack slices with ...[begin:end:step]:\n";

  std::cout<< "  Non-type template parameters:\n";
  std::cout<< "    "<< x...[::-1]<<"\n" ...;

  std::cout<< "  Type template parameters:\n";
  std::cout<< "    "<< @type_string(types_t...[::-1])<< "\n" ...;

  std::cout<< "  Function template parameters:\n";
  std::cout<< "    "<< args...[::-1]<< "\n" ...;
}

int main() {
  func1<100, 200, 300>(4, 5l, 6ll);
  func2<100, 200, 300>(4, 5l, 6ll);
  func3<100, 200, 300>(4, 5l, 6ll);
}