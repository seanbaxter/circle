#include <cstdio>
#include <iostream>
#include <tuple>

template<typename type_t>
void func(const type_t& obj) {
  inspect(obj) {
    // Try to convert a 3-element tuple-like:
    // First element is convert to short.
    // Second element is dereferenced and converted to float.
    // Third element is dereferenced and converted to double.
    tuple as [short, *float, **double] => {
      std::cout<< decltype(tuple).string<< ": ";
      (std::cout<< tuple...[:]<< " " ...)<< "\n";
    }

    // Follow * by _ to indicate wildcard. We don't try to convert to anything,
    // we're only testing that an element is a pointer and non-null.
    is [_, *_, **_] => std::cout<< type_t.string + ": passes pointer test\n";
    
    is _            => std::cout<< type_t.string + ": no match\n";
  }
}

int main() {
  int x = 10;
  int y = 100; 
  int z = 1000; int* pz = &z;

  struct obj_t { int x, *y, **z; };
  obj_t obj { x, &y, &pz };
  func(obj);
  
  // Create a [value, pointer, pointer-to-pointer] of non-arithmetic types.
  obj_t* pobj = &obj;
  auto tuple = std::make_tuple(obj, &obj, &pobj);
  func(tuple);

  // Clear the middle pointer to nullptr.
  get<1>(tuple) = nullptr;
  func(tuple);
}