#include <vector>
#include <cstdio>

int main() {
  // Loop over all odd indices and break when i > 10.
  for(int i : @range(1::2)...) {
    printf("%d ", i);
    if(i > 10)
      break;
  }
  printf("\n");
  
  // The same as above, but put the end index in the range.
  for(int i : @range(1:10:2)...)
    printf("%d ", i);
  printf("\n");

  int items[] { 5, 2, 2, 3, 1, 0, 9, 8 };

  // Loop over all but the first item.
  for(int i : items[1:]...)
    printf("%d ", i);
  printf("\n");

  // Loop over items in reverse order.
  for(int i : items[::-1]...)
    printf("%d ", i);
  printf("\n");

  // Bind to the range expression which adds consecutive elements.
  // The items array has 8 elements, but this loop runs through 7 elements,
  // because the slice expression items[1:] starts at index 1 (so only has
  // 7 elements).
  for(int x : items[:] + items[1:]...)
    printf("%d ", x);
  printf("\n");
}