#include <cuda_runtime.h>
#include <cstdio>
#include <typeinfo>

struct base_t {
  base_t(int x) : x(x) { }
  virtual ~base_t() { }
  virtual void func() = 0;

  int x;
};

struct derived1_t : base_t {
  derived1_t(int x) : base_t(x) { }
  void func() override {
    printf("Hello\n");
  }
};

struct derived2_t : base_t {
  derived2_t(int x) : base_t(x) { }
  void func() override {
    printf("Goodbye\n");
  }
};

struct derived3_t : derived1_t, derived2_t {
  derived3_t(int x1, int x2) : derived1_t(x1), derived2_t(x2) { }
  void func() override {
    printf("Hello and Goodbye\n");
  }
};

__global__ void kernels() {
  base_t* b = new derived1_t(1);
  b->func();

  // Get the typeid of the dynamic type using RTTI.
  printf("%s\n", typeid(*b).name());

  // dynamic_cast up to derived1_t.
  auto* d1 = dynamic_cast<derived1_t*>(b);
  auto* d2 = dynamic_cast<derived2_t*>(b);
  printf("b=%p  d1=%p d2=%p\n", b, d1, d2);

  // Invoke the deleting destructor. This calls a hidden second destructor
  // on base_t, which is overridden by the dynamic type derived1_t.
  delete b;

  // Support dynamic-cast across derived types.
  d1 = new derived3_t(3, 4);
  d2 = dynamic_cast<derived2_t*>(d1);
  printf("%s: d1=%p d2=%p\n", typeid(*d1).name(), d1, d2);
  printf("d1.x = %d, d2.x = %d\n", d1->x, d2->x);
  delete d1;
}

int main() {
  kernels<<<1,1>>>();
  cudaDeviceSynchronize();
}