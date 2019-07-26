#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdarg>

// Use popen to make a system call and capture the output in a file handle.
// Make it inline to prevent it from being emitted by the backend, unless
// called by real code.

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
