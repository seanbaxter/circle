#include <cstdlib>
#include <cstdio>
#include <ctime>

// Use popen to make a system call and capture the output in a file handle.
// Make it inline to prevent it from being output by the backend.
inline int capture_call(const char* cmd, char* text, int len) {
  FILE* f = popen(cmd, "r");
  len = f ? fread(text, 1, len, f) : 0;
  pclose(f);
}

// Every time print_version is compiled, it runs "git rev-parse HEAD" to
// get the current commit hash.
void print_version() {
  // Make a format specifier to print the first 10 digits of the git hash.
  @meta const char* fmt =
    "  Circle compiler\n"
    "  Written by Sean Baxter\n"
    "  Compiled %s"
    "  Build %d\n"
    "  Hash: %.10s\n";

  // Get the time of compilation.
  @meta time_t t = time(0);
  @meta const char* time = asctime(gmtime(&t));

  // Retrieve the current commit hash. The hash is 40 digits long, and we
  // include space for null.
  @meta char hash[41];
  @meta int len = capture_call("git rev-parse HEAD", hash, 41);

  // Substitute into the format specifier.
  @meta char text[120];
  @meta sprintf(text, fmt, time, __circle_build__, hash);
  
  // The text array has automatic storage duration at *compile time*. The
  // array will expire when the end of the function is reached, so it will be
  // inaccessible at runtime, which is when we want to print the message.
  // Use @string to convert the compile-time data to a string literal which 
  // is available at runtime.
  puts(@string(text));
}

int main() {
  print_version();
  return 0;
}
