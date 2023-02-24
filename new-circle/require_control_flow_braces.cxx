#feature on require_control_flow_braces

int main() {
  int x = 0;

  if(1) { 
    ++x;          // OK.
  }

  if(1)           // Error.
    ++x;

  for(int i = 0; i < 10; ++i) {
    x *= 2;       // OK
  }

  for(int i = 0; i < 10; ++i)
    x *= 2;       // Error.

  while(false) {
    x *= 3;       // OK
  }

  while(false)
    x *= 3;       // Error.
}