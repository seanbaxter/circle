#include <cstdio>

void duff_copy2(char* dest, const char* source, size_t count) {
  const char* end = source + count;
  while(size_t count = end - source) {
    switch(count % 8) {
      @meta for(int i = 8; i > 0; --i)
        case i % 8: *dest++ = *source++;
    
      break;
    }
  }
}

int main(int argc, char** argv) {
  return 0;
}