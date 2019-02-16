#include <cstdlib>

template<size_t N>
void duff_copy3(char* dest, const char* source, size_t count) {
  static_assert(N > 0, "argument N must be > 0");

  const char* end = source + count; 
  while(size_t count = end - source) {
    switch(count % N) {
      @meta for(int i = N; i > 0; --i)
        // Expand inside the non-meta switch
        case i % N: *dest++ = *source++;
  
      break;
    }
  }
}

void test8(char* dest, const char* source, size_t count) {
  duff_copy3<8>(dest, source, count);
}

void test16(char* dest, const char* source, size_t count) {
  duff_copy3<16>(dest, source, count);
}

int main(int argc, char** argv) {
  return 0;
}