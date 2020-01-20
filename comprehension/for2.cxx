#include <vector>
#include <map>
#include <string>
#include <cstdio>

int main() {
  std::map<std::string, int> map {
    { "Washington", 1 },
    { "Madison", 5 },
    { "Lincoln", 16 },
    { "Grant", 18 },
    { "Coolidge", 30 },
  };

  std::string s = [for &x : map => x.first[:]... ...];
  puts(s.c_str());
}


