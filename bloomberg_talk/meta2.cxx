#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

inline std::string capture_call(const char* cmd) {
  // We want to capture both stdout (for nominal operation) and stderr
  // (for failure message). Append a stderr->stdout pipe.
  std::string cmd2 = cmd + std::string(" 2>&1");
  FILE* f = popen(cmd2.data(), "r");

  // We can't seek within this stream, so read in a KB at a time.
  std::vector<char> vec;
  char line[1024];
  size_t count = 0;
  do {
    count = fread(line, 1, 1024, f);
    vec.insert(vec.end(), line, line + count);

  } while(1024 == count);

  std::string s(vec.begin(), vec.end());
  
  // If the process closes with a non-zero exit code, throw the terminal
  // output as a std::runtime_error exception.
  if(pclose(f))
    throw std::runtime_error(s);

  return s;
}

// Every time print_version is compiled, it runs "git rev-parse HEAD" to
// get the current commit hash.
void print_version() {
  // Make a format specifier to print the first 10 digits of the git hash.
  @meta const char* fmt =
    "  Circle compiler\n"
    "  2019 Sean Baxter\n"
    "  hash: %.10s\n";

  // Retrieve the current commit hash.
  @meta std::string hash = capture_call("git rev-parse HEAD");

  // Substitute into the format specifier.
  @meta char text[120];
  @meta sprintf(text, fmt, hash.c_str());

  // Convert to a string literal and print.
  puts(@string(text));
}

int main() {
  print_version();
  return 0;
}

