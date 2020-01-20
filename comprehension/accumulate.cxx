// Sums the first ten squares and prints them, using views::ints to generate
// and infinite range of integers, views::transform to square them, views::take
// to drop all but the first 10, and accumulate to sum them.

#include <vector>
#include <iostream>

int main() {
  auto sq = [](int x) { return x * x; };
  int sum = (... + sq(@range(1:11)));
  std::cout<< sum<< '\n';   // Prints 385
}