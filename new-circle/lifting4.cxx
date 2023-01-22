#include <ranges>
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
  // {3, 0} should come AFTER {3, 1}, because we're only comparing the
  // get<0> element, not a full tuple comparison.
  std::vector<std::tuple<int, int> > v{{3,1},{2,4},{1,7},{3,0}};

  // Use the lifting lambda []std::get<0>. This is invoked internally by
  // ranges::sort to extract the 0th element from each tuple, for
  // comparison purposes.
  std::ranges::sort(v, std::less{}, []std::get<0>);

  for(auto obj : v) {
    std::cout<< get<0>(obj)<< ", "<< get<1>(obj)<< "\n";
  }
}