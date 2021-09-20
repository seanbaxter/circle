#include <cuda_runtime.h>
#include <set>
#include <cstdio>

__global__ void kernel(int count, int range) {
  // Generate count number of random numbers on the device.
  // Feed them through std::set. This means just one of each number
  // is kept, and duplicates are removed.
  std::set<int> set;
  for(int i : count)
    set.insert(rand() % range);

  // Print the unique, sorted elements to the terminal.
  printf("%d unique values generated:\n", set.size());
  int index = 0;
  for(int x : set)
    printf("%2d: %3d\n", index++, x);
}

int main() {
  kernel<<<1, 1>>>(10, 10);
  cudaDeviceSynchronize();
}