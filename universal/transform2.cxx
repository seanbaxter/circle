#include <iostream>
#include <tuple>
#include <utility>
#include <algorithm>

// Re-arrange the tuple by element size.
// Elements with the smallest size are sorted to go first in result object.
auto sort_tuple(const auto& tuple) {
  // Sort once per template instantiation. .first is the sizeof the element.
  // .second is the gather index.
  @meta std::pair<int, int> sizes[] { 
    std::make_pair(sizeof(tuple.[:]), int...) ...
  };
  @meta std::sort(sizes + 0, sizes + sizeof. sizes); 

  // The gather operation. ...[] gathers from tuple. sizes...[:].second is the
  // gather index for each output.
  return std::make_tuple(tuple.[sizes.[:].second] ...);
}

int main() {  
  auto tuple = std::make_tuple(1, 2.f, '3', 4ll, 5.0, 6);
  auto tuple2 = sort_tuple(tuple);

  std::cout<< decltype(tuple2).string << "\n";
  std::cout<< tuple2.[:]<< " (size = "<< sizeof(tuple2.[:])<< ")\n" ...;
}