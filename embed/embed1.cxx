#include <cstdio>

inline const char* filename = "test_binary.data";
const int data[] = @embed(int, filename);

@meta printf("data has %zu bytes\n", sizeof(data));

int main() {
  // Use it or lose it.
  for(int x : data)
    printf("%d\n", x);

  return 0;
}



