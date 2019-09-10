#include <tuple>
#include <map>
#include <string>

void cxx17_structured_bindings() {

  // Structured binding (positional) to public non-static data members x, y, z.
  struct foo_t {
    int x, y, z;
  };  
  foo_t obj1 { 5, 6, 7 };
  auto& [a1, b1, c1] = obj1;
  printf("%d %d %d\n", a1, b1, c1);
  
  // Structured binding on tuple-like object.
  std::tuple<int, double, const char*> obj2 {
    10,
    3.14,
    "a very long string"
  };
  auto& [a2, b2, c2] = obj2;
  printf("%d %f %s\n", a2, b2, c2);

  // Structured binding on array.
  int array3[] { 10, 20, 30, 40 };  
  auto [a3, b3, c3, d3] = array3;
  printf("%d %d %d %d\n", a3, b3, c3, d3);

  // Structured binding in a ranged-for loop. Each element of map is an 
  // std::pair, which is "tuple-like" by C++'s definitinon. The structured
  // binding uses std::get<0> and std::get<1> to decompose the pair into the
  // [key, value] declarations.
  std::map<int, std::string> map {
    { 1, "One" }, { 2, "Two" }, { 3, "Three" }
  };
  for(auto& [key, value] : map)
    printf("%d : %s\n", key, value.c_str());
}

int main() {
  cxx17_structured_bindings();
  return 0;
}