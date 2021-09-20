#include <algorithm>
#include <utility>
#include <string>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// Convert ints to roman numerals.
// http://rosettacode.org/wiki/Roman_numerals/Encode#C.2B.2B
inline std::string to_roman(int value) {
  struct romandata_t { int value; char const* numeral; };
  static romandata_t const romandata[] =
     { 1000, "M",
        900, "CM",
        500, "D",
        400, "CD",
        100, "C",
         90, "XC",
         50, "L",
         40, "XL",
         10, "X",
          9, "IX",
          5, "V",
          4, "IV",
          1, "I",
          0, NULL }; // end marker
 
  std::string result;
  for (romandata_t const* current = romandata; current->value > 0; ++current)
  {
    while (value >= current->value)
    {
      result += current->numeral;
      value  -= current->value;
    }
  }
  return result;
}
 
__global__ void test_vector(int count) {
  // Generate count random integers.
  std::vector<std::pair<std::string, int> > romans(count);
  for(int i : count) {
    // Don't know how to count higher than 4000.
    int x = rand() % 4000;
    romans[i] = std::make_pair(to_roman(x), x);
  }

  // Now sort them according to lexicographical order of their
  // roman numerals. 
  std::sort(romans.begin(), romans.end());

  // Print the sorted roman numerals.
  printf("%4d - %s\n", romans[:].second, romans[:].first.c_str())...;
}

int main() {
  test_vector<<<1, 1>>>(20);
  cudaDeviceSynchronize();
}
