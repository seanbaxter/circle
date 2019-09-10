#include <cstdio>

template<typename... types_t>
void func(types_t... params) {
  
  @meta @mtype x = @dynamic_type(decltype(params...[5]));

}

int main() {
  func(1, 2, 3, 4, 5);
  return 0;
}