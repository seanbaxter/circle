#include <vector>
#include <iostream>
#include <cassert>

template<typename It, typename F>
auto find_extreme(It begin, It end, F f) {
  assert(begin != end);

  auto x = *begin++;
  while(begin != end)
    x = f(x, *begin++);

  return x;
}

int main() {
  std::vector<int> vec { 10, 4, 7, 19, 14, 3, 2, 11, 14, 15 };

  // Pass a lifting lambda for the max and min function templates.
  auto max = find_extreme(vec.begin(), vec.end(), []std::max);
  auto min = find_extreme(vec.begin(), vec.end(), []std::min);

  std::cout<< "min is "<< min<< "\n";
  std::cout<< "max is "<< max<< "\n";
}
 