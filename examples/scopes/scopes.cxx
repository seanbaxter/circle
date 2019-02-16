// circle -filetype=ll 

#include <cstdio>

// Meta statements work in global scope.
@meta printf("%s:%d I'm in global scope.\n", __FILE__, __LINE__);

namespace ns {
  // Or in namespace scope.
  @meta printf("%s:%d Hello namespace.\n", __FILE__, __LINE__);

  struct foo_t {
    // Also in class definitions.
    @meta printf("%s:%d In a class definition!\n", __FILE__, __LINE__);

    enum my_enum {
      // Don't forget enums.
      @meta printf("%s:%d I'm in your enum.\n", __FILE__, __LINE__);
    };

    void func() const {
      // And naturally in function/block scope.
      // Ordinary name lookup finds __func__ in this function's 
      // declarative region.
      @meta printf("%s ... And block scope.\n", __func__);
    }
  };
}

int main() {
  return 0;
}