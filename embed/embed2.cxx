#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <vector>
#include <stdexcept>

inline std::string format(const char* pattern, ...) {
  va_list args;
  va_start(args, pattern);

  va_list args_copy;
  va_copy(args_copy, args);

  int len = std::vsnprintf(nullptr, 0, pattern, args);
  std::string result(len, ' ');
  std::vsnprintf((char*)result.data(), len + 1, pattern, args_copy);

  va_end(args_copy);
  va_end(args);

  return result;
}

template<typename type_t>
std::vector<type_t> read_file(const char* filename) {
  FILE* f = fopen(filename, "r");
  if(!f) {
    throw std::runtime_error(format("could not open the file %s:%s", filename));
  }

  // Should really use stat here.
  fseek(f, 0, SEEK_END);
  size_t length = ftell(f);
  fseek(f, 0, SEEK_SET);

  // 128 MB sounds like a lot.
  const size_t max_length = 128<< 20;
  if(length > max_length) {
    throw std::runtime_error(
      format("file %s has length %zu; max allowed is %zu", filename, length,
        max_length));
  }

  // File size must be divisible by type size.
  if(length % sizeof(type_t)) {
    throw std::runtime_error(
      format("file %s has length %zu which is not divisible by size of %s (%zu)",
        filename, length, @type_string(type_t), sizeof(type_t))
    );
  }

  // Read the file in.
  size_t count = length / sizeof(type_t);

  // Size the vector once.
  std::vector<type_t> storage(count);

  // Read the data.
  fread(storage.data(), sizeof(type_t), count, f);

  // Close the file.
  fclose(f);

  // Return the file.
  return std::move(storage);
}

// Read the file.
@meta puts("Reading the file from disk");
@meta auto file_data = read_file<int>("test_binary.data");

// Inject into an array.
@meta puts("Injecting into an array with @array");
const int data[] = @array(file_data.data(), file_data.size());

int main() {

  for(int x : data)
    printf("%d\n", x);

  return 0;
}





