#include <cstdio>

enum class my_enum {
  a, b;                       // Declare a, b

  // Loop from 'c' until 'f'
  @meta for(char x = 'c'; x != 'f'; ++x) {
    // Get a string ready.
    @meta char name[2] = { x, 0 };

    // Declare the enum.
    @(name);
  }

  f, g;                       // Declare f, g
};

// Prints a b c d e f g
@meta for(int i = 0; i < @enum_count(my_enum); ++i)
  @meta printf("%s ", @enum_name(my_enum, i));
@meta printf("\n");

int main(int argc, char** argv) { 
  return 0; 
}