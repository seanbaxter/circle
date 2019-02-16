#include <cstdlib>

void duff_copy1(char* dest, const char* source, size_t count) {
  const char* end = source + count;
  while(size_t count = end - source) {
    switch(count % 8) {
      case 0: *dest++ = *source++; // Fall-through to case 7
      case 7: *dest++ = *source++; // Fall-through to case 6...
      case 6: *dest++ = *source++;
      case 5: *dest++ = *source++;
      case 4: *dest++ = *source++;
      case 3: *dest++ = *source++;
      case 2: *dest++ = *source++;
      case 1: *dest++ = *source++;
      break;
    }
  }
}

int main(int argc, char** argv) {
  return 0;
}
