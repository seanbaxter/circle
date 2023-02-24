#feature on interface self forward template_brackets
#include <iostream>
#include <string>
#include <vector>
#include <memory>

// make_type_erased is like std::make_unique, but it returns a 
// *type erased* unique_ptr. There are two explicit template parameters:
// 1. The type to allocate on the heap.
// 2. The interface of the return dyn type.
// This is library code. Write it once, file it away.
template<typename Type, interface IFace>
std::unique_ptr!dyn!IFace make_unique_dyn(forward auto... args) {
  return std::unique_ptr!dyn!IFace(make_dyn<IFace>(new Type(forward args...)));
}

// Define an interface. This serves as our "abstract base class".
interface IPrint {
  void print() const;
};

// Implement for arithmetic types.
template<typename T> requires(T~is_arithmetic)
impl T : IPrint {
  void print() const {
    std::cout<< T~string + ": "<< self<< "\n";
  }
};

// Implement for string.
impl std::string : IPrint {
  void print() const {
    std::cout<< "std::string: "<< self<< "\n";
  }
};

// Implement for a container of type-erased types.
impl std::vector!std::unique_ptr!dyn!IPrint : IPrint {
  void print() const {
    std::cout<< "std::vector!std::unique_ptr!dyn!IPrint:\n";
    for(const auto& obj : self) {
      // Loop through all elements. Print out 2 spaces to indent.
      std::cout<< "  ";

      // Invoke the type-erased print function.
      obj->print();
    }
  }
};

int main() {
  std::vector!std::unique_ptr!dyn!IPrint vec;

  // Allocate and a push an unsigned short : IPrint;
  vec.push_back(make_unique_dyn!<unsigned short, IPrint>(2));

  // Allocate and push an int : IPrint.
  vec.push_back(make_unique_dyn!<int, IPrint>(5));

  // Allocate and push a double : IPrint.
  vec.push_back(make_unique_dyn!<double, IPrint>(3.14));

  // Allocate and push a string : IPrint.
  vec.push_back(make_unique_dyn!<std::string, IPrint>("Hello type erasure"));

  // Loop over all elements and call the print() interface method.
  // This is a homogeneous, type-erased interface for heterogeneous data.
  vec.IPrint::print();

  // When vec goes out of scope, its destructor calls unique_ptr's destructor,
  // and that calls the dyn-deleting destructor stored in the dyntable of
  // each type. For types with trivial destructors, this is just the
  // deallocation function.
  // *All resources get cleaned up*.
}