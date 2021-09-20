#include <vector>
#include <cstdio>
#include <cuda_runtime.h>

// Declare the storage of an object. Don't execute its ctor or dtor!
[[storage_only]] __device__ std::vector<int> global_vec;

__global__ void global_ctors() {
  // Launch a single-thread dynamic initializer kernel.
  // Use placement new to construct global_vec.
  new (&global_vec) std::vector<int>();
}

__global__ void global_dtors() {
  // Call the pseudo-destructor to destruct global_vec.
  global_vec.~vector();
}

__global__ void load() {
  // Push 10 items 
  for(int i : 10)
    global_vec.push_back(i * i);
}

__global__ void print() {
  // Each thread prints one vector item.
  int tid = threadIdx.x;
  if(tid < global_vec.size())
    printf("%3d: %3d\n", tid, global_vec[tid]);
}

int main() {
  // Inititialize.
  global_ctors<<<1,1>>>();

  // Do the work.
  load<<<1, 1>>>();
  print<<<1, 128>>>();
  cudaDeviceSynchronize();

  // Cleanup.
  global_dtors<<<1,1>>>();
}