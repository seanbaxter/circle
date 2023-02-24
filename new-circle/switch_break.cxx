#include <iostream>
#feature on switch_break

int main() {
  for(int arg = 1; arg < 8; ++arg) {
    int x = 0;
    switch(arg) {
      case 1:
      case 2:
        x = 100;
        // implicit break here.

      case 3:
      case 4:
        x = 200;
        [[fallthrough]];  // Use the fallthrough attribute from C++17.
                          // Effectively "goto case 5"

      case 5:
      case 6:
        x = 300;

        // Support conditional fallthroughs, as long as the next statement
        // is a label.
        if(6 == arg)
          [[fallthrough]];  // Effectively "goto default".

      // implicit break here. It stops arg==5, but not arg==6.
      default:
        x = 400;
    }

    std::cout<< arg<< " -> "<< x<< "\n";
  }
}