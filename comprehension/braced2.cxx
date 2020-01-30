#include <string>
#include <map>
#include <cstdio>

int main() {

  // Create two compile-time maps.
  @meta std::map<std::string, int> map1 {
    { "alpha",    5 },
    { "gamma",   10 },
    { "epsilon", 20 }
  };

  @meta std::map<std::string, int> map2 {
    { "beta",    15 },
    { "omega",    4 },
    { "iota",    19 }
  };

  // Merge map2 into map1 at compile time.
  @meta map1[map2[:].first] = map2[:].second ...;

  // Print the merged map at compile time.
  @meta printf("%-20s: %d\n", map1[:].first.c_str(), map1[:].second)...;

  // Port this to a runtime map.
  std::map<std::string, int> map3 {
    std::make_pair(@string(map1[:].first), (int)map1[:].second) ...
  };

  // Print all pairs in the runtime map.
  printf("%-20s: %d\n", map3[:].first.c_str(), map3[:].second)...;
}
