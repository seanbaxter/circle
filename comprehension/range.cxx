#include <vector>
#include <string>
#include <cstdio>

int sq(int x) { return x * x; }

int main() {
  printf("%d ", @range(10))...; 
  printf("\n");        // Prints '0 1 2 3 4 5 6 7 8 9 '

  printf("%d ", @range(5:25:5))...;
  printf("\n");        // Prints '5 10 15 20 '

  printf("%d ", @range(25:5:-5))...;
  printf("\n");        // Prints '24 19 14 9'

  // Sum up integers from 0 to 9.
  int sum = (... + @range(10));

  // Sum up squares of integers from 0 to 9.
  int sum_squares = (... + sq(@range(10)));

  // Fill two vectors with ints.
  std::vector v1 = [@range(3:18:3)...];   // 3, 6, 9, 12, 15
  std::vector v2 = [@range(5:15:2)...];   // 5, 7, 9, 11, 13

  printf("%d ", v1[:])...; printf("\n");
  printf("%d ", v2[:])...; printf("\n");

  // Get their dot product.
  double dot = (... + (v1[:] * v2[:]));
  printf("%f\n", dot);

  // Get their L2 norm.
  double l2 = sqrt(... + sq(v1[:] - v2[:]));
  printf("%f\n", l2);

  // Fill a vector with strings.
  const char* days[] {
    "Sunday", "Monday", "Tuesday", "Wednesday", 
    "Thursday", "Friday", "Saturday"
  };
  // Print index/string pairs.
  printf("%d: %s\n", @range(1:), days[:])...;

  // Prints:
  // 1: Sunday
  // 2: Monday
  // 3: Tuesday
  // 4: Wednesday
  // 5: Thursday
  // 6: Friday
  // 7: Saturday
}
