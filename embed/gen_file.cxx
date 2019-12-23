#include <cstdio>
#include <cstdlib>
#include <vector>

const int size = 50; //12'500'000;

int main() {
  std::vector<int> data(size);
  for(int i = 0; i < size; ++i)
    data[i] = 1000 * i + i;

  FILE* f = fopen("test_binary.data", "w");
  fwrite(data.data(), sizeof(int), data.size(), f);
  fclose(f);

  return 0;
}

