#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>

__global__ void kernel(int count) {
  std::vector<int> data(count);

  // Generate random numbers between 0 and 9.
  for(int i : count)
    data[i] = rand() % 10;

  // Sort with qsort.
  auto cmp = [](const void* a, const void* b) {
    int x = *(const int*)a;
    int y = *(const int*)b;

    // Return a 3-way comparison. 
    return x < y ? -1 : x > y ? +1 : 0;
  };
  qsort(data.data(), data.size(), sizeof(int), cmp);

  printf("%2d: %2d\n", @range(), data[:])...;

  // Binary search for the first occurrence of 4 in the sorted array.
  int key = 4;
  if(void* p = bsearch(&key, data.data(), count, sizeof(int), cmp)) {
    int index = (int*)p - data.data();
    printf("The first occurrence of %d is index %d\n", key, index);

  } else {
    printf("No occurrence of %d\n", key);
  }
}

int main() {
  kernel<<<1,1>>>(20);
  cudaDeviceSynchronize();
}