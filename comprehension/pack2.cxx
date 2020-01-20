#include <vector>
#include <algorithm>
#include <cstdio>

inline int sq(int x) {
  return x * x;
}

int main() {
  // std::vector = [ @range(10)... ];
  std::vector<int> x;
  {
    // The expansion count is inferred from the range expression.
    size_t count = 10;

    // Declare an object for the current index in the range.
    int begin = 0;

    // Loop until we've exhausted the range.
    while(count--) {
      x.push_back(begin);

      // The begin index for positive step is inclusive. Decrement it at
      // the end of the loop.
      ++begin;
    }
  }

  // std::vector y = [ @range(10::-1)... ];
  std::vector<int> y;
  {
    size_t count = 10;
    int begin = 10;
    while(count--) {
      // The begin index for negative step is exclusive. Decrement it at
      // the start of the loop.
      --begin;

      y.push_back(begin);
    }
  }

  // std::vector z = [ sq(x[:]) + 5 * y[:] ... ];
  std::vector<int> z;
  {
    // Find the size for each slice in the pack expression.
    size_t x_count = x.size();
    size_t y_count = y.size();

    // Use the minimum slice size to set the loop count.
    size_t count = std::min(x_count, y_count);

    // Declare iterators for the current item in each slice.
    auto x_begin = x.begin();
    auto y_begin = y.begin();

    while(count--) {
      z.push_back(sq(*x_begin) + 5 * *y_begin);

      // Both slices have an implicit +1 step, so perform post-step increment.
      ++x_begin;
      ++y_begin;
    }
  }

  // printf("sq(%d) + 5 * %d -> %2d\n", x[:], y[:], z[:]) ...;
  {
    // Find the size for each slice in the pack expression.
    size_t x_count = x.size();
    size_t y_count = y.size();
    size_t z_count = z.size();
    size_t count = std::min(std::min(x_count, y_count), z_count);

    auto x_begin = x.begin();
    auto y_begin = y.begin();
    auto z_begin = z.begin();

    while(count--) {
      printf("sq(%d) + 5 * %d -> %2d\n", *x_begin, *y_begin, *z_begin);

      ++x_begin;
      ++y_begin;
      ++z_begin;
    }
  }
}
