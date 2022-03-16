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
  derived1_t(int x) : base_t(x) { 
    printf("  derived1_t::derived1_t() constructor called\n");
  }
  ~derived1_t() {
    printf("  derived1_t::~derived1_t() destructor called\n");
  }
  void func() override {
    printf("  derived1_t::func() called: Hello\n");
  }
};

struct derived2_t : base_t {
  derived2_t(int x) : base_t(x) { 
    printf("  derived2_t::derived2_t() constructor called\n");
  }
  ~derived2_t() {
    printf("  derived2_t::~derived2_t() destructor called\n");
  }
  void func() override {
    printf("  derived2_t::func() called: Goodbye\n");
  }
};

struct derived3_t : derived1_t, derived2_t {
  derived3_t(int x1, int x2) : derived1_t(x1), derived2_t(x2) { 
    printf("  derived3_t::derived3_t() constructor called\n");
  }
  ~derived3_t() {
    printf("  derived3_t::~derived3_t() destructor called\n");
  }
  void func() override {
    printf("  derived3_t::func() called: Hello and Goodbye\n");
  }
};

__global__ void kernel() {
  printf("kernel: creating derived1_t and upcasting to base_t*:\n");
  base_t* base = new derived1_t(1);
  base->func();

  // Get the typeid of the dynamic type using RTTI.
  printf("\nkernel: typeid of the base_t: %s\n", typeid(*base).name());

  // dynamic_cast up to derived1_t.
  printf("\nkernel: dynamic_cast from base_t:\n");
  derived1_t* derived1 = dynamic_cast<derived1_t*>(base);
  printf("  dynamic_cast<derived1_t>(base) = %p\n", derived1);
  derived2_t* derived2 = dynamic_cast<derived2_t*>(base);
  printf("  dynamic_cast<derived2_t>(base) = %p\n", derived2);

  // Invoke the deleting destructor. This calls a hidden second destructor
  // on base_t, which is overridden by the dynamic type derived1_t.
  printf("\nkernel: calling the deleting dtor on base:\n");
  delete base;

  // Support dynamic-cast across derived types.
  printf("\nkernel: creating derived3_t and upcasting to derived1_t:\n");
  derived1 = new derived3_t(3, 4);

  printf("\nkernel: typeof of the derived1_t: %s\n", typeid(*derived1).name());

  printf("\nkernel: dynamic_cast from derived1_t:\n");
  derived2 = dynamic_cast<derived2_t*>(derived1);
  printf("  dynamic_cast<derived2_t>(derived1) = %p\n", derived2);
  derived3_t* derived3 = dynamic_cast<derived3_t*>(derived1);
  printf("  dynamic_cast<derived3_t>(derived1) = %p\n", derived3);

  printf("\nkernel: calling the deleting dtor on derived1:\n");
  delete derived1;
}

int main() {
  kernel<<<1,1>>>();
  cudaDeviceSynchronize();
}